from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import Optional

import torch
from omegaconf import DictConfig
from torch import nn
from torch import Tensor
from transformers.models.bert.modeling_bert import BertPreTrainedModel

from fz_openqa.modeling.modules.utils.backbone import instantiate_backbone_model_with_config
from fz_openqa.utils.metric_type import MetricType


class Head(nn.Module, ABC):
    """Score question and document representations."""

    def __init__(
        self,
        *,
        bert: DictConfig | BertPreTrainedModel,
        output_size: int,
        metric_type: MetricType = MetricType.inner_product,
        id: str = "base",
        **kwargs,
    ):
        super(Head, self).__init__()
        self.register_buffer("_temperature", torch.tensor(-1.0))
        self.register_buffer("_offset", torch.tensor(0.0))
        self.max_seq_length = bert.config.max_length
        self.id = id
        self.metric_type = MetricType(metric_type)

        # instantiate backbone
        bert = instantiate_backbone_model_with_config(bert)

        # input and output dimensions
        self.input_size = bert.config.hidden_size
        self.output_size = output_size

    @property
    def requires_external_backbone(self) -> bool:
        if not hasattr(self, "backbone"):
            return True
        else:
            return self.backbone is None

    @property
    def temperature(self) -> Tensor:
        return self._temperature

    @property
    def offset(self):
        return self._offset

    def kl(self) -> Tensor | float:
        return 0.0

    def entropy(self) -> Tensor | float:
        return 0.0

    @abstractmethod
    def forward(
        self,
        *,
        hd: Tensor,
        hq: Tensor,
        doc_ids: Optional[Tensor] = None,
        q_mask: Optional[Tensor] = None,
        d_mask: Optional[Tensor] = None,
        batch: Dict[str, Tensor] = None,
        **kwargs,
    ) -> Dict:
        """
        Compute the score for each pair `f([q_j; a_j], d_jk)`.

        Parameters
        ----------
        hd
            Document representations of shape [bs, n_opts, n_docs, ...]
        hq
            Question representations of shape [bs, n_opts, ...]
        doc_ids
            Document ids of shape [bs, n_opts, n_docs]
        q_mask
            Mask for the question representations
        d_mask
            Mask for the document representations
        batch
            Batch of data
        Returns
        -------
        Tensor
            Dict, containing the scores of shape [bs, n_opts, n_docs]
        """
        raise NotImplementedError

    def preprocess(self, last_hidden_state: Tensor, head: str, **kwargs) -> Tensor:
        return self._preprocess(last_hidden_state, head, **kwargs)

    def _preprocess(
        self, last_hidden_state: Tensor, head: str, mask: Optional[Tensor] = None, **kwargs
    ) -> Tensor:
        return last_hidden_state
