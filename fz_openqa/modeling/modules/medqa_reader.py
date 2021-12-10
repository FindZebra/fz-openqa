import warnings
from typing import Any
from typing import List
from typing import Optional
from typing import Union

import rich
import torch
from datasets import Split
from torch import Tensor
from transformers import AutoTokenizer

from ...datamodules.pipes import PrintBatch
from ...utils.pretty import pprint_batch
from .base import Module
from .utils import expand_and_flatten
from .utils import flatten_first_dims
from fz_openqa.modeling.functional import flatten
from fz_openqa.modeling.functional import padless_cat
from fz_openqa.utils.datastruct import Batch


class MedQaReader(Module):
    # name of the features required for a forward pass
    _required_feature_names = [
        "answer.target",
        "question.input_ids",
        "question.attention_mask",
        "document.input_ids",
        "document.attention_mask",
    ]

    # named of the features required for evaluation
    _required_eval_feature_names = ["answer.target", "document.match_score"]

    # prefix for the logged metrics
    task_id: Optional[str] = "reader"

    # metrics to display
    pbar_metrics = [
        "train/reader/Accuracy",
        "validation/reader/Accuracy",
        "train/reader/relevance-Accuracy",
        "validation/reader/relevance-Accuracy",
    ]

    _required_heads = ["option", "evidence", "relevance"]

    def _init_metrics(self, prefix: str = ""):
        """Initialize a Metric for each split=train/validation/test
        fir both the answering model and the selection model"""
        self.answer_metrics = self._get_base_metrics(prefix=prefix)

    def _forward(self, batch: Batch, targets, **kwargs) -> Batch:
        # tokenizer = AutoTokenizer.from_pretrained(self.bert.name_or_path, use_fast=True)
        # checks inputs, set parameters and concat the questions with the documents
        pprint_batch(batch)
        # self.bert.tokenizer
        # concatenate questions and documents such that there is no padding between Q and D
        qd_batch = self._concat_questions_and_documents(batch, fields=["question", "document"])
        # pprint_batch(qd_batch)
        # tokenizer = AutoTokenizer.from_pretrained(self.bert.name_or_path, use_fast=True)
        rich.print(f"[cyan] ANS: {self.tokenizer.encode('[ANS]')}")
        rich.print(f"[cyan] QUERY: {self.tokenizer.encode('[QUERY]')}")
        rich.print(f"[cyan] DOC: {self.tokenizer.encode('[DOC]')}")
        rich.print(
            f"[red] {[qd_batch['input_ids'][0][i].tolist() for i in range(4)]}"
        )  # noqa: E501
        rich.print(
            f"[red] {[self.tokenizer.decode(qd_batch['input_ids'][0][i].tolist()) for i in range(4)]}"  # noqa: E501
        )
        exit()
        # rich.print(f"[red] {answer_targets.shape}")
        # pprint_batch(qd_batch)

        return self.bert(**qd_batch, labels=targets, return_dict=True)

    def _step(self, batch: Batch, **kwargs: Any) -> Batch:

        # compute the reader loss
        answer_targets: Tensor = batch["answer.target"]

        # forward pass through the reader model
        outputs = self._forward(batch, targets=answer_targets, **kwargs)

        loss, logits = outputs[:2]

        return {
            "loss": loss,
            "_logits_": logits.detach(),
            "_answer_targets_": answer_targets.detach(),
        }

    def _reduce_step_output(self, output: Batch) -> Batch:
        """
        Gather losses from all devides and return
        """

        # average losses
        for k in ["loss"]:
            y = output.get(k, None)
            if y is not None:
                output[k] = y.mean()

        return output

    def _concat_questions_and_documents(self, batch: Batch, *, fields: List[str]):
        """
        Concatenate fields across the time dimension, and without padding
        """
        assert set(fields) == {
            "question",
            "document",
        }, "Question and document fields must be provided."
        bs, n_options, *_ = batch["question.input_ids"].shape
        rich.print(bs, n_options)
        batch = self._flatten_qd(batch)

        inputs = [
            {
                "input_ids": batch[f"{key}.input_ids"].view(
                    -1, *batch[f"{key}.input_ids"].shape[2:]
                )[:, 1:],
                "attention_mask": batch[f"{key}.attention_mask"].view(
                    -1, *batch[f"{key}.attention_mask"].shape[2:]
                )[:, 1:],
            }
            for key in fields
        ]

        # concatenate keys across the time dimension, CLS tokens are removed
        padded_batch = padless_cat(
            inputs,
            master_key="input_ids",
            pad_token=self._pad_token_id,
            aux_pad_tokens={"attention_mask": 0},
        )
        # append the CLS tokens
        for key in ["input_ids", "attention_mask"]:
            cls_tokens = batch[f"{fields[0]}.{key}"][:, :, :1]
            padded_batch[key] = padded_batch[key].view(bs, n_options, *padded_batch[key].shape[1:])
            padded_batch[key] = torch.cat([cls_tokens, padded_batch[key]], dim=-1)

        # length of concatenated fields
        input_length = padded_batch["input_ids"].shape[-1]

        if input_length > self.max_length:
            warnings.warn(f"the tensor [{'; '.join(fields)}] was truncated.")
            for key in ["input_ids", "attention_mask"]:
                padded_batch[key] = padded_batch[key][:, :, : self.max_length]

        return padded_batch

    @staticmethod
    def _flatten_qd(batch: Batch) -> Batch:
        # flatten docs of shape [bs, n_options, n_docs, L_docs] to [bs, n_options, n_docs * L_docs]
        keys = ["document.input_ids", "document.attention_mask"]
        d_batch = {k: batch[k][:, :, :, 1:] for k in keys}
        for key in keys:
            cls_tokens = batch[key][:, :, 0, :1].squeeze(0)
            d_batch[key] = d_batch[key].contiguous().view(*d_batch[key].shape[:-2], -1)
            d_batch[key] = torch.cat([cls_tokens, d_batch[key]], dim=-1)

        # join question features and document features
        keys = ["question.input_ids", "question.attention_mask"]
        q_batch = {k: batch[k] for k in keys}

        return {**q_batch, **d_batch}

    def update_metrics(self, output: Batch, split: Split) -> None:
        """update the metrics of the given split."""
        answer_logits, answer_targets = (
            output.get(k, None) for k in ("_logits_", "_answer_targets_")
        )
        self.answer_metrics.update(split, answer_logits, answer_targets)

    def reset_metrics(self, split: Optional[Split] = None) -> None:
        """
        Reset the metrics corresponding to `split` if provided, else
        reset all the metrics.
        """
        self.answer_metrics.reset(split)

    def compute_metrics(self, split: Optional[Split] = None) -> Batch:
        """
        Compute the metrics for the given `split` else compute the metrics for all splits.
        The metrics are return after computation.
        """
        return {
            **self.answer_metrics.compute(split),
        }
