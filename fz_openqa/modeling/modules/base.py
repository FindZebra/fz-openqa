from __future__ import annotations

import collections
import warnings
from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import torch
import torch.nn.functional as F
from datasets import Split
from omegaconf import DictConfig
from torch import nn
from torch import Tensor
from torchmetrics.classification import Accuracy
from transformers import BertPreTrainedModel
from transformers import PreTrainedTokenizerFast
from transformers.models.bert.modeling_bert import BertEncoder

from fz_openqa.modeling.modules.utils.bert import instantiate_bert_model_with_config
from fz_openqa.modeling.modules.utils.metrics import SafeMetricCollection
from fz_openqa.modeling.modules.utils.metrics import SplitMetrics
from fz_openqa.tokenizers.static import ANS_TOKEN
from fz_openqa.tokenizers.static import DOC_TOKEN
from fz_openqa.tokenizers.static import QUERY_MASK
from fz_openqa.tokenizers.static import QUERY_TOKEN
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.functional import batch_reduce
from fz_openqa.utils.functional import maybe_instantiate
from fz_openqa.utils.pretty import pprint_batch


def is_feature_name(x):
    return str(x).startswith("_") and str(x).endswith("_")


class Module(nn.Module, ABC):
    """
    A model:
        1. computes the loss
        2. computes and track the metrics (accuracy, F1, ...) using `SplitMetrics`

    !! Important !!
    Metrics needs to be updated in the call of `_step_end` in the
    LightningModule in order to avoid errors with dp.
    Therefore all the update steps need to be implemented in `update_metrics`,
    which is subsequently called in
    BaseModel.step_end() on device 0.
    See https://torchmetrics.readthedocs.io/en/stable/pages/
        overview.html#metrics-in-dataparallel-dp-mode
    """

    _vocabulary_size: int
    _pad_token_id: int

    # name of the features required for a forward pass
    _required_feature_names = [
        "input_ids",
        "attention_mask",
    ]

    # named of the features required for evaluation
    _required_eval_feature_names = [
        "input_ids",
        "attention_mask",
        "labels",
    ]

    # prefix for the logged metrics
    task_id: Optional[str] = None

    # metrics to display
    pbar_metrics = ["train/loss", "train/Accuracy", "validation/Accuracy"]

    # maximum input size
    max_length = 512  # todo: infer

    def __init__(
        self,
        *,
        bert: DictConfig | BertPreTrainedModel,
        tokenizer: DictConfig | PreTrainedTokenizerFast,
        split_bert_layers: int = 0,
        prefix: str = "",
        **kwargs,
    ):
        """Initialize a Metric for each split=train/validation/test"""
        super().__init__()
        self.tokenizer: PreTrainedTokenizerFast = maybe_instantiate(tokenizer)
        self.bert: BertPreTrainedModel = self._instantiate_bert(
            bert=bert, tokenizer=self.tokenizer, split_bert_layers=split_bert_layers
        )
        self._init_metrics(prefix=prefix)

    def _backbone(
        self,
        batch: Batch,
        prefix: Optional[str] = None,
        max_batch_size: Optional[int] = None,
        **kwargs,
    ) -> Union[Tensor, Dict[str, Tensor]]:

        # select the keys with prefix
        if prefix is not None:
            batch = self._select(prefix, batch)

        if batch["input_ids"].shape[1] > self.max_length:
            warnings.warn(
                f"In model {type(self).__name__}, "
                f"truncating input_ids with "
                f"length={batch['input_ids'].shape[1]} > max_length={self.max_length}"
            )

        # get input data
        inputs_ids = batch["input_ids"][:, : self.max_length]
        attention_mask = batch["attention_mask"][:, : self.max_length]

        # process data by chunk
        output = None
        bs, seq_len, *_ = inputs_ids.shape
        if max_batch_size is None:
            chunk_size = bs
        else:
            chunk_size = int(max_batch_size * (self.max_length / seq_len) ** 2)

        for i in range(0, bs, chunk_size):
            chunk = self._process_tokens(
                inputs_ids[i : i + chunk_size],
                attention_mask[i : i + chunk_size],
                **kwargs,
            )
            if output is None:
                output = chunk
            else:
                output = torch.cat([output, chunk], dim=0)

        return output

    def _process_tokens(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        **kwargs,
    ) -> Tensor:
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return bert_output.last_hidden_state

    def _instantiate_bert(
        self,
        *,
        bert: Union[BertPreTrainedModel, DictConfig],
        tokenizer: PreTrainedTokenizerFast,
        split_bert_layers: int = 0,
    ) -> BertPreTrainedModel:
        """Instantiate a bert model, and extend its embeddings to match the tokenizer"""

        self._vocabulary_size = len(tokenizer.get_vocab())
        self._pad_token_id = tokenizer.pad_token_id

        # instantiate the bert model using `bert.config`
        bert = instantiate_bert_model_with_config(bert)

        # drop the last `drop_bert_layers` layers
        if split_bert_layers > 0:
            encoder: BertEncoder = bert.encoder
            encoder.layer = nn.ModuleList(encoder.layer[:-split_bert_layers])

        # extend BERT embeddings for the added special tokens
        if bert.get_input_embeddings().weight.shape[0] != len(tokenizer):
            bert.resize_token_embeddings(len(tokenizer))

            emb_map = {
                QUERY_MASK: tokenizer.pad_token_id,
                QUERY_TOKEN: tokenizer.sep_token_id,
                DOC_TOKEN: tokenizer.sep_token_id,
                ANS_TOKEN: tokenizer.sep_token_id,
            }

            # embs = bert.get_input_embeddings()
            self.spec_tokens = {}
            for token, target_idx in emb_map.items():
                tok_id = tokenizer.encode(token, add_special_tokens=False)[0]
                self.spec_tokens[token] = tok_id
                # embs.weight.data[tok_id] = embs.weight.data[target_idx]
            # bert.set_input_embeddings(embs)
            # bert.tie_weights()

            # e = bert.get_input_embeddings().weight
            # q_mask_id = tokenizer.encode(QUERY_MASK, add_special_tokens=False)[0]
            # doc_id = tokenizer.encode(DOC_TOKEN, add_special_tokens=False)[0]
            # assert (e[q_mask_id] == e[tokenizer.pad_token_id]).all()
            # print(f"embs[q_mask_id] = {e[q_mask_id].mean()}")
            # print(f"embs[pad_token_id] = {e[tokenizer.pad_token_id].mean()}")
            # print(f"embs[doc_id] = {e[doc_id].mean()}")
            # print(f"embs[sep_token_id] = {e[tokenizer.sep_token_id].mean()}")

        return bert

    def forward(self, batch: Batch, **kwargs):
        """Compute the forward pass of the model, does not require targets,
        it can be used for inference."""
        self._check_features(batch, self._required_feature_names)
        return self._forward(batch, **kwargs)

    def evaluate(self, batch: Batch, filter_features=False, **kwargs):
        """
        Evaluate the model (step + step end) given a batch of data
        with targets
        """
        step_output = self.step(batch, **kwargs)
        return self.step_end(
            step_output, None, update_metrics=False, filter_features=filter_features
        )

    def step(self, batch: Batch, **kwargs: Any) -> Batch:
        """Compute the forward pass of the model and return output
        Return a dictionary output with at least the key 'loss' and the data
        necessary to compute the metrics, unless the loss is explicitly
        computed in the `post_forward` method.

        This step will be computed in the `*_step()` method of the
        ligthning module: the data is processed separately on each device.

        The torchmetric `Metric.update()` method should not be called here.
        See `post_forward` instead.

        Implement `_step` for each sub-class.
        """
        pprint_batch(batch, "Module: inpput.step")
        self._check_features(batch, self._required_eval_feature_names)
        return self._step(batch, **kwargs)

    def step_end(
        self,
        output: Batch,
        split: Optional[Split],
        update_metrics: bool = True,
        filter_features: bool = True,
    ) -> Any:
        """Apply a post-processing step to the forward method.
        The output is the output of the forward method.

        This method is called after the `output` has been gathered
        from each device. This method must aggregate the loss across
        devices.

        torchmetrics update() calls should be placed here.
        The output must at least contains the `loss` key.

        Implement `_reduce_step_output` for each sub-class.
        """

        # reduce tensors gathered from parallel `step()` calls
        output = self._reduce_step_output(output)

        # update the metrics
        if update_metrics:
            assert split is not None
            self.update_metrics(output, split)

        # filter internal values (e.g. __targets__
        if filter_features:
            output = self._filter_features_from_output(output)
        return output

    @abstractmethod
    def _forward(self, batch: Batch, **kwargs):
        return self.backbone(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            **kwargs,
        )

    @abstractmethod
    def _step(self, batch: Batch, **kwargs: Any) -> Batch:
        """Run the forward pass and potentially compute the loss.
        A loss that requires outputs from all device must be
        implemented in `_reduce_step_output`"""
        logits = self.forward(batch, **kwargs)
        nll = F._batched_cross_entropy(logits, batch["labels"], reduce="none")

        # register internal values (which should not be passed to the pl module) using _<name>_.
        return {
            "loss": batch_reduce(nll, op=torch.mean),
            "_logits_": logits.detach(),
            "_targets_": batch["labels"].detach(),
        }

    @abstractmethod
    def _reduce_step_output(self, step_output: Batch) -> Batch:
        """Compute values based on the tensors gathered form each value"""
        step_output["loss"] = step_output["loss"].mean()
        return step_output

    def update_metrics(self, output: Batch, split: Split) -> None:
        """update the metrics of the given split."""
        logits, targets = (output[k] for k in ("_logits_", "_targets_"))
        self.reader_metrics.update(split, logits, targets)

    def reset_metrics(self, split: Optional[Split] = None) -> None:
        """
        Reset the metrics corresponding to `split` if provided, else
        reset all the metrics.
        """
        self.reader_metrics.reset(split)

    def compute_metrics(self, split: Optional[Split] = None) -> Batch:
        """
        Compute the metrics for the given `split` else compute the metrics for all splits.
        The metrics are return after computation.
        """
        return self.reader_metrics.compute(split)

    @staticmethod
    def _get_base_metrics(
        *,
        prefix: Optional[str] = None,
        metric_kwargs: Optional[Dict] = None,
        topk: Optional[List[Union[None, int]]] = None,
    ) -> SplitMetrics:
        """
        Return the base metrics for the given prefix.

        Parameters
        ----------
        prefix
            The prefix of the metrics group
        metric_kwargs
            The kwargs to pass to the metrics
        topk
            The list of topk to compute (None means default Accuracy)
        Returns
        SplitMetrics
            the metrics group
        -------

        """
        metric_kwargs = metric_kwargs or {"compute_on_step": False, "dist_sync_on_step": True}

        if topk is None:
            topk = [None]

        def _name(k):
            return f"top{k}_Accuracy" if k is not None else "Accuracy"

        metrics = SafeMetricCollection(
            {_name(k): Accuracy(top_k=k, **metric_kwargs) for k in topk},
            prefix=prefix,
        )

        return SplitMetrics(metrics)

    def _init_metrics(self, prefix: str):
        self.reader_metrics = self._get_base_metrics(prefix=prefix)

    @staticmethod
    def _filter_features_from_output(output: Batch) -> Batch:
        """filter the internal values such as _logits_ or _targets_"""
        return {k: v for k, v in output.items() if not is_feature_name(k)}

    @staticmethod
    def _select(prefix: str, batch: Batch) -> Batch:
        """Select attributes with prefix `prefix` from the `batch`"""
        prefix = f"{prefix}."
        return {k.replace(prefix, ""): v for k, v in batch.items() if str(k).startswith(prefix)}

    def _check_batch_type(self, batch: Batch) -> None:
        """
        Check that the batch input is of the right type.
        Potentially raise an error.
        """
        assert isinstance(batch, (dict, collections.OrderedDict, collections.UserDict))

    def _format_exception(self, batch: Batch, required_feature_names: List[str]) -> str:
        missing = [f for f in required_feature_names if f not in batch]
        return (
            f"{type(self).__name__} requires features {required_feature_names}, "
            f"features={missing} are missing in batch with keys={list(batch.keys())}."
        )

    def _check_features(self, batch, required_feature_names: List[str]):
        if any(f not in batch for f in required_feature_names):
            raise ValueError(self._format_exception(batch, required_feature_names))
