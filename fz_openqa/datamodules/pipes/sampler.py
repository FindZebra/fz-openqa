from __future__ import annotations

import math
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Union

import datasets
import rich
import torch
from datasets import Split
from loguru import logger
from omegaconf import DictConfig
from pydantic import BaseModel
from pydantic import PositiveFloat
from pydantic import PositiveInt
from torch import Tensor
from warp_pipes import Batch
from warp_pipes import Pipe
from warp_pipes.support.tensor_handler import TensorFormat
from warp_pipes.support.tensor_handler import TensorHandler


class SamplerConfig(BaseModel):
    total: Union[PositiveInt, Dict[str, PositiveInt]] = 10
    temperature: Union[PositiveFloat, Dict[str, PositiveFloat]] = 1.0


class PrioritySamplerConfig(SamplerConfig):
    mode: Literal["exponential", "uniform"] = "uniform"


class Sampler(Pipe):
    ConfigCls = SamplerConfig

    def __init__(
        self,
        *,
        config: Optional[SamplerConfig] = None,
        proposal_score_key: str = "proposal_score",
        idx_key: str = "row_idx",
        log_weight_key: str = "proposal_log_weight",
        field="document",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.proposal_score_key = f"{field}.{proposal_score_key}"
        self.idx_key = f"{field}.{idx_key}"
        self.log_weight_key = f"{field}.{log_weight_key}"
        self.field = field
        if isinstance(config, (DictConfig, dict)):
            config = self.ConfigCls(**config)
        elif config is None:
            config = self.ConfigCls()
        self.config = config

    def _infer_split_config(self, split: datasets.Split) -> SamplerConfig:
        output_config = {}
        for key, value in self.config.dict().items():
            if isinstance(value, dict):
                output_config[key] = value[split]
            else:
                output_config[key] = value
        return self.ConfigCls(**output_config)

    def _call_batch(
        self, batch: Batch, idx: Optional[List[int]] = None, split: Split = None, **kwargs
    ) -> Batch:
        config = self._infer_split_config(split)
        handler = TensorHandler(TensorFormat.TORCH)

        logits = handler(batch[self.proposal_score_key])
        indices = handler(batch[self.idx_key])

        # sample
        with torch.no_grad():
            ids, log_w = self.sample(logits=logits, config=config)

        # re-index and return
        logits = logits.gather(-1, index=ids)
        indices = indices.gather(-1, index=ids)

        return {
            self.proposal_score_key: logits,
            self.idx_key: indices,
            self.log_weight_key: log_w,
        }

    @staticmethod
    def sample(
        logits: torch.Tensor,
        config: SamplerConfig,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if config.temperature < 0:
            raise ValueError("temperature must be non-positive")

        if config.temperature == 0:
            indices = torch.argsort(logits, descending=True)
            indices = indices[..., : config.total]
        else:
            indices = logits.multinomial(num_samples=config.total, dim=-1, replacement=True)

        # uniform importance weights
        log_w = torch.zeros_like(logits[..., : config.total])
        log_w -= math.log(config.total)
        return indices, log_w


class PrioritySampler(Sampler):
    """Sample using `priority sampling`: https://arxiv.org/abs/cs/0509026"""

    ConfigCls = PrioritySamplerConfig
    MAX_LOG_RANGE: float = 1e5

    def __init__(self, *args, mode: Literal["uniform", "exponential"] = "uniform", **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = mode

    @staticmethod
    def sample(
        logits: torch.Tensor,
        config: PrioritySamplerConfig,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(config, PrioritySamplerConfig)

        logits_ = logits.clone()
        logits = PrioritySampler.clip_logits(logits, dim=-1)
        log_pz = logits.log_softmax(dim=-1)

        if config.mode == "uniform":
            if config.temperature == 0:
                u = 0.5 * torch.ones_like(log_pz)
            else:
                u = torch.rand_like(log_pz).clamp(min=1e-12)
        elif config.mode == "exponential":
            if config.temperature == 0:
                u = torch.ones_like(log_pz)
            else:
                u = torch.empty_like(log_pz)
                u.exponential_()
        else:
            raise ValueError(f"Unknown mode {config.mode}")

        log_u = u.log()
        keys = log_pz - log_u
        z = keys.argsort(dim=-1, descending=True)[..., : config.total + 1]
        if config.total < logits.shape[-1]:
            z_tau = z[..., -1:]
            log_tau = keys.gather(dim=-1, index=z_tau)[..., :1]
        else:
            log_tau = -float("inf") + torch.zeros_like(log_pz)
        z = z[..., : config.total]
        log_pz = log_pz.gather(dim=-1, index=z)

        if config.mode == "uniform":
            log_qz = torch.where(log_pz - log_tau < 0, log_pz - log_tau, torch.zeros_like(log_pz))
        elif config.mode == "exponential":
            log_qz = log_pz - log_tau
            log_qz = (log_qz).exp().mul(-1).exp().mul(-1).log1p()
        else:
            raise ValueError(f"Unknown mode {config.mode}")

        # warn if NaNs are found
        n_nans = log_qz.isnan().float().sum()
        if n_nans > 0:
            logger.warning(f"Found {n_nans} NaNs in log_qz")
            rich.print(f">> log_qz: {log_qz}\nlogits_:{logits_}")
        n_nans = log_pz.isnan().float().sum()
        if n_nans > 0:
            logger.warning(f"Found {n_nans} NaNs in log_pz")
            rich.print(f">> log_pz: {log_pz}\nlogits_:{logits_}")

        # finally, compute the log weights
        log_weight = log_pz - log_qz

        return z, log_weight

    @staticmethod
    def clip_logits(logits: Tensor, dim=-1):
        mask_pos_inf = logits.isinf() & (logits > 0)
        non_inf_values = logits[(~logits.isinf()) & (~logits.isnan())]
        if non_inf_values.numel() == 0:
            logger.error("All logits are inf or -inf, setting to logits to zero")
            return logits.zero_()

        # handle the positive and negative infs
        if mask_pos_inf.float().sum() > 0:
            # positive inf values are not allowed, but handle it for now
            logger.error(
                f"Found {mask_pos_inf.float().sum()} positive infs in logits, "
                f"replacing with max value. "
                f"TODO: investigate the origin of the +infs"
            )
            logits.masked_fill_(mask_pos_inf, non_inf_values.max())

        # scale the logits
        M = logits.max(dim=dim, keepdim=True).values
        logits = logits - M

        # clip log_pz for numerical stability
        output = M + logits.clamp(min=-PrioritySampler.MAX_LOG_RANGE)

        return output
