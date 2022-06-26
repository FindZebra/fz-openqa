from __future__ import annotations

import os
import string
from typing import Any
from typing import Dict
from typing import Optional

import rich
import torch
from datasets import Split
from loguru import logger
from omegaconf import DictConfig
from torch import Tensor

from ...datamodules.pipes import PrintContent
from ..heads.dpr import DprHead
from ..heads.dpr import unique_with_indices
from .utils.total_epoch_metric import TotalEpochMetric
from .utils.utils import flatten_first_dims
from fz_openqa.modeling.gradients import Gradients
from fz_openqa.modeling.gradients import ReinforceGradients
from fz_openqa.modeling.heads.base import Head
from fz_openqa.modeling.modules.base import Module
from fz_openqa.modeling.modules.utils.metrics import SafeMetricCollection
from fz_openqa.modeling.modules.utils.metrics import SplitMetrics
from fz_openqa.utils import maybe_instantiate
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.fingerprint import fingerprint_bert
from fz_openqa.utils.fingerprint import get_fingerprint
from fz_openqa.utils.pretty import pprint_batch

VERBOSE_MODEL = bool(os.environ.get("VERBOSE_MODEL", False))


class ReaderRetriever(Module):
    """
    A reader-retriever model for OpenQA.
    """

    _required_feature_names = []

    _required_eval_feature_names = [
        "question.input_ids",
        "question.attention_mask",
        "document.input_ids",
        "document.attention_mask",
    ]

    # prefix for the logged metrics
    task_id: Optional[str] = None

    # metrics to display in the progress bar
    pbar_metrics = [
        "train/reader/logp",
        "validation/reader/logp",
        "train/reader/Accuracy",
        "validation/reader/Accuracy",
        "train/retrieval/MRR",
        "validation/retrieval/MRR",
    ]

    def __init__(
        self,
        *args,
        reader_head: None | Head | DictConfig,
        retriever_head: Head | DictConfig,
        mask_punctuation: bool = False,
        max_batch_size: Optional[int] = None,
        gradients: Gradients | DictConfig = None,
        **kwargs,
    ):
        if gradients is None:
            gradients = ReinforceGradients()

        super().__init__(*args, **kwargs)

        # register the heads
        self.reader_head: Optional[Head] = maybe_instantiate(reader_head)
        self.retriever_head: Head = maybe_instantiate(retriever_head)

        # parameters
        self.max_batch_size = max_batch_size

        # init the estimator
        self.estimator = maybe_instantiate(gradients)

        # punctuation masking
        self.skiplist = []
        if mask_punctuation:
            self.skiplist += [
                self.tokenizer.encode(symbol, add_special_tokens=False)[0]
                for symbol in string.punctuation
            ]
        self.register_buffer("sep_token_id", torch.tensor(self.tokenizer.sep_token_id))
        self.register_buffer("pad_token_id", torch.tensor(self.tokenizer.pad_token_id))

        # Log the fingerprint of the model
        for k, hs in fingerprint_bert(self.backbone).items():
            logger.info(f"Fingerprint:backbone:{k}={hs}")
        logger.info(f"Fingerprint:head:reader={get_fingerprint(self.reader_head)}")
        logger.info(f"Fingerprint:head:retriever={get_fingerprint(self.retriever_head)}")

    def _forward(self, batch: Batch, head: None | str = "retriever", **kwargs) -> Batch:
        """
        Compute a forward pass through
            1. the shared backbone model (BERT, GPT, ...)
            2. if `head` is provided, forward through the selected head.
        Parameters
        ----------
        batch
            input data
        head
            id of the head to process (reader / retriever), None means skipping the head
        kwargs
            kwargs passed to `_forward_backbone`
        Returns
        -------
        Dict
            A dictionary containing the features for each `field` and `head`
        """
        output = {}
        field_names = {"question", "document"}
        heads = {"reader": self.reader_head, "retriever": self.retriever_head}

        # process all available fields
        for field_name in field_names:
            if f"{field_name}.input_ids" in batch:

                # process the `input_ids` using the backbone.yaml (BERT)
                h = self._forward_backbone(batch, field_name, **kwargs)

                # process `h` using the head(s)
                if head is not None:
                    mask = self.get_mask(batch, field_name)
                    for hid in head.split("+"):
                        head_layer = heads[hid]
                        if head_layer is None:
                            continue
                        h_ = head_layer._preprocess(h, field_name, mask=mask, batch=batch)
                        feature_name = self._get_feature_name(field=field_name, head=hid)
                        output[feature_name] = h_
                else:
                    feature_name = self._get_feature_name(field=field_name, head=None)
                    output[feature_name] = h

        return output

    def _forward_backbone(
        self,
        batch: Batch,
        field: str,
        silent: bool = True,
        max_batch_size: Optional[int] = None,
        **kwargs,
    ) -> Tensor:
        """
        Forward pass through the shared BERT model.
        Parameters
        ----------
        batch
            input data
        field
            field to process (`question` or `document`)
        silent
            deactivate logging
        max_batch_size
            maximum batch size used to process the BERT model (see `_backbone()`)
        kwargs
            additional arguments passed to `_backbone()`
        Returns
        Tensor
            output features
        -------

        """
        original_shape = batch[f"{field}.input_ids"].shape
        pprint_batch(batch, f"forward {field}", silent=silent)

        # set the batch size allowed for one forward pass,
        # if the batch size exceeds the limit, the model processes the data by chunk
        if max_batch_size is None:
            max_batch_size = self.max_batch_size
        elif max_batch_size < 0:
            max_batch_size = None

        # flatten the batch
        flat_batch = flatten_first_dims(
            batch,
            n_dims=len(original_shape) - 1,
            keys=[f"{field}.input_ids", f"{field}.attention_mask", f"{field}.token_type_ids"],
        )

        # process the document with the backbone.yaml
        h = self._backbone(
            flat_batch,
            prefix=f"{field}",
            max_batch_size=max_batch_size,
            **kwargs,
        )

        # reshape and return
        h = h.view(*original_shape[:-1], *h.shape[1:])
        return h

    def _step(self, batch: Batch, silent=not VERBOSE_MODEL, **kwargs: Any) -> Batch:
        """
        Compute the forward pass for the question and the documents.
        """
        pprint_batch(batch, "Option retriever::Input::batch", silent=silent)

        # initialize output dict
        step_output = {}

        # process the batch using BERT
        max_batch_size_eval = -1 if self.training else self.max_batch_size
        features = self._forward(
            batch, head=None, silent=silent, max_batch_size=max_batch_size_eval, **kwargs
        )
        pprint_batch(features, "Option retriever::BERT+heads::features", silent=silent)

        # compute the scores using the heads
        head_meta = {}
        head_kwargs = {
            "q_mask": self.get_mask(batch, "question"),
            "d_mask": self.get_mask(batch, "document"),
            "doc_ids": batch["document.row_idx"],
            "batch": batch,
            **kwargs,
        }
        if self.reader_head is not None:
            reader_meta = self.reader_head(
                hd=features[self._get_feature_name(field="document", head=None)],
                hq=features[self._get_feature_name(field="question", head=None)],
                **head_kwargs,
            )
            head_meta.update({f"reader_{k}": v for k, v in reader_meta.items()})
        if self.retriever_head is not None:
            retriever_meta = self.retriever_head(
                hd=features[self._get_feature_name(field="document", head=None)],
                hq=features[self._get_feature_name(field="question", head=None)],
                **head_kwargs,
            )
            head_meta.update({f"retriever_{k}": v for k, v in retriever_meta.items()})

        # gather all outputs
        step_output = {
            **step_output,
            **head_meta,
            **kwargs,
            **{k: batch[k] for k in self.estimator.required_features},
        }

        pprint_batch(step_output, "Option retriever::step::output", silent=silent)

        return step_output

    def _reduce_step_output(self, step_output: Batch) -> Batch:
        """
        Gather losses and logits from all devices and return
        """

        pprint_batch(step_output, "reduce_step_output::input", silent=not VERBOSE_MODEL)
        PrintContent(
            keys=[
                "question.id",
                "question.loc",
                "question.target",
            ]
        )(step_output)

        # process with the estimator
        output = self.estimator(**step_output)

        pprint_batch(output, "reduce_step_output::estimator::output", silent=not VERBOSE_MODEL)

        # add the head diagnostics
        output.update(self._get_heads_diagnostics())

        # average losses
        for k, v in output.items():
            if not str(k).startswith("_") and not str(k).endswith("_"):
                if isinstance(v, torch.Tensor):
                    v = v.float().mean()
                output[k] = v

        return output

    def _get_heads_diagnostics(self):
        diagnostics = {}
        if self.reader_head is not None:
            diagnostics["reader/temperature"] = self.reader_head.temperature.detach()
        if self.retriever_head is not None:
            diagnostics["retriever/temperature"] = self.retriever_head.temperature.detach()
        return diagnostics

    @staticmethod
    def _expand_docs_to_options(x: Tensor, doc_target_shape: Optional[torch.Size]) -> Tensor:
        if doc_target_shape is None:
            return x
        elif x.shape[: len(doc_target_shape)] != doc_target_shape:
            x = x.unsqueeze(1)
            x = x.expand(*doc_target_shape[:2], *x.shape[2:])
            return x
        else:
            return x

    @torch.no_grad()
    def update_metrics(self, output: Batch, split: Split) -> None:
        """update the metrics of the given split."""

        # log the reader accuracy
        reader_logits = output.get("_reader_logits_", None)
        reader_targets = output.get("_reader_targets_", None)
        retriever_reading_logits = output.get("_retriever_reading_logits_", None)
        if reader_targets is not None:
            if reader_logits is not None:
                self.reader_metrics.update(split, reader_logits, reader_targets)
            if retriever_reading_logits is not None:
                self.retriever_reading_metrics.update(
                    split, retriever_reading_logits, reader_targets
                )

        # log the retriever accuracy
        retriever_logits = output.get("_retriever_logits_", None)
        retriever_targets = output.get("_retriever_binary_targets_", None)
        if retriever_logits is not None and retriever_logits.numel() > 0:
            if retriever_targets is not None:
                self.retriever_metrics.update(split, retriever_logits, retriever_targets)

    def reset_metrics(self, split: Optional[Split] = None) -> None:
        """
        Reset the metrics corresponding to `split` if provided, else
        reset all the metrics.
        """
        self.reader_metrics.reset(split)
        self.retriever_reading_metrics.reset(split)
        self.retriever_metrics.reset(split)

    def compute_metrics(self, split: Optional[Split] = None) -> Batch:
        """
        Compute the metrics for the given `split` else compute the metrics for all splits.
        The metrics are return after computation.
        """
        return {
            **self.reader_metrics.compute(split),
            **self.retriever_reading_metrics.compute(split),
            **self.retriever_metrics.compute(split),
        }

    def step_end(
        self,
        output: Batch,
        split: Optional[Split],
        update_metrics: bool = True,
        filter_features: bool = True,
    ) -> Any:
        """
        Force returning features during validation: required by `LogPredictionsCallback`
        """
        if split is not None and not split == Split.TRAIN:
            filter_features = False

        return super().step_end(
            output, split, filter_features=filter_features, update_metrics=update_metrics
        )

    def _init_metrics(self, prefix: str):
        """Initialize the metrics for each split."""
        self.reader_metrics = self._get_base_metrics(prefix=f"{prefix}reader/")
        self.retriever_reading_metrics = self._get_base_metrics(prefix=f"{prefix}reader/retriever-")

        self.retriever_metrics = self._get_base_metrics(
            prefix=f"{prefix}retrieval/",
            topk=[1, 3, 5, 10, 20, 50, 100],
            retrieval=True,
            allowed_splits=[Split.VALIDATION, Split.TEST],
        )

    def get_mask(self, batch: Batch, field: str) -> Tensor:
        """Build the mask for the specified field"""
        if field == "question":
            mask = batch["question.attention_mask"].clone().float()
            mask[mask == self._pad_token_id] = 0
        elif field == "document":
            inputs_ids = batch["document.input_ids"]
            mask = torch.ones_like(batch["document.attention_mask"].float())
            mask[mask == self._pad_token_id] = 0
            for w in self.skiplist:
                mask[inputs_ids == w] = 0
        else:
            raise ValueError(f"Unknown field {field}")

        return mask

    def _get_feature_name(self, *, field: str, head: Optional[str] = None) -> str:
        field_prefixes = {"document": "_hd_", "question": "_hq_"}
        head_suffixes = {"retriever": "", "reader": "reader_", None: "last_hidden_state_"}
        field_prefix = field_prefixes[field]
        head_suffix = head_suffixes[head]
        return f"{field_prefix}{head_suffix}"
