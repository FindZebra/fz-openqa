import warnings
from typing import Any
from typing import List
from typing import Optional
from typing import Union

import rich
import torch
from datasets import Split
from torch import Tensor

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

    # def _init_metrics(self, prefix: str = ""):
    #     """Initialize a Metric for each split=train/validation/test
    #     fir both the answering model and the selection model"""
    #     self.answer_metrics = self._get_base_metrics(prefix=prefix)

    def _forward(self, batch: Batch, **kwargs) -> Batch:

        # checks inputs, set parameters and concat the questions with the documents
        bs, n_options, *_ = batch["answer.input_ids"].shape
        _, n_docs, *_ = batch["document.input_ids"].shape

        assert n_options == 4
        pprint_batch(batch)

        # concatenate questions and documents such that there is no padding between Q and D
        qd_batch = self._concat_questions_and_documents(
            batch, fields=["answer", "question", "document"]
        )
        pprint_batch(qd_batch)
        exit()
        # qd_batch = {k: v.view(-1, qd_batch.size(-1)) for k, v in qd_batch.items()}

        # compute the reader loss
        answer_targets: Tensor = batch["answer.target"]

        test_batch = {k: v.unsqueeze(0) for k, v in qd_batch.items()}
        rich.print(f"[red] {answer_targets.shape}")
        pprint_batch(qd_batch)
        pprint_batch(test_batch)

        return self.bert(**qd_batch, labels=answer_targets, return_dict=True)

    def _step(self, batch: Batch, **kwargs: Any) -> Batch:

        # forward pass through the reader model
        outputs = self._forward(batch, **kwargs)

        loss, logits = outputs[:2]

        return {
            "loss": loss,
            "_logits_": logits.detach(),
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

    def _concat_questions_and_answers(self, batch: Batch, *, fields: List[str]):
        """
        Concatenate fields across the time dimension, and without padding
        """
        assert set(fields) == {
            "question",
            "answer",
        }, "Question and document fields must be provided."
        batch_size, n_options = batch["answer.input_ids"].shape[:2]

        # expand and flatten documents and questions such that
        # both are of shape [batch_size * n_docs, ...]
        batch = self._expand_and_flatten_qa(batch, n_options)

        # concatenate keys across the time dimension, CLS tokens are removed
        padded_batch = padless_cat(
            [
                {
                    "input_ids": batch[f"{key}.input_ids"][:, 1:],
                    "attention_mask": batch[f"{key}.attention_mask"][:, 1:],
                }
                for key in fields
            ],
            master_key="input_ids",
            pad_token=self._pad_token_id,
            aux_pad_tokens={"attention_mask": 0},
        )

        # append the CLS tokens
        for key in ["input_ids", "attention_mask"]:
            cls_tokens = batch[f"{fields[0]}.{key}"][:, :1]
            padded_batch[key] = torch.cat([cls_tokens, padded_batch[key]], 1)

        if len(padded_batch["input_ids"]) > self.max_length:
            warnings.warn(f"the tensor [{'; '.join(fields)}] was truncated.")

        return padded_batch

    def _concat_questions_and_documents(self, batch: Batch, *, fields: List[str]):
        """
        Concatenate fields across the time dimension, and without padding
        """
        assert set(fields) == {
            "question",
            "document",
            "answer",
        }, "Question and document fields must be provided."
        batch_size, n_docs = batch["document.input_ids"].shape[:2]

        #
        # batch = self._flatten_qd(batch, n_)
        pprint_batch(batch)
        # concatenate keys across the time dimension, CLS tokens are removed
        padded_batch = padless_cat(
            [
                {
                    "input_ids": batch[f"{key}.input_ids"][:, 1:],
                    "attention_mask": batch[f"{key}.attention_mask"][:, 1:],
                }
                for key in fields
            ],
            master_key="input_ids",
            pad_token=self._pad_token_id,
            aux_pad_tokens={"attention_mask": 0},
        )

        # append the CLS tokens
        for key in ["input_ids", "attention_mask"]:
            cls_tokens = batch[f"{fields[0]}.{key}"][:, :1]
            padded_batch[key] = torch.cat([cls_tokens, padded_batch[key]], 1)

        # length of concatenated fields
        input_length = padded_batch["input_ids"].shape[-1]
        pprint_batch(padded_batch)

        if input_length > self.max_length:
            warnings.warn(f"the tensor [{'; '.join(fields)}] was truncated.")
            for key in ["input_ids", "attention_mask"]:
                padded_batch[key] = padded_batch[key][:, : self.max_length]

        return padded_batch

    def _expand_and_flatten_qa(self, batch: Batch, n_options: int) -> Batch:
        # flatten documents of shape [bs, n_docs, L_docs] to [bs*n_docs, L_docs]
        d_batch = flatten_first_dims(
            batch,
            2,
            keys=["answer.input_ids", "answer.attention_mask"],
        )
        # expand questions to shape [bs, n_docs, L_1] and flatten to shape [bs*n_docs, L_q]
        q_batch = expand_and_flatten(
            batch,
            n_options,
            keys=["question.input_ids", "question.attention_mask"],
        )
        return {**q_batch, **d_batch}

    def _flatten_qd(self, batch: Batch) -> Batch:
        # flatten documents of shape [bs, n_docs, L_docs] to [bs, n_docs * L_docs]
        keys = ["document.input_ids", "document.attention_mask"]
        d_batch = {k: batch[k].view(batch[k].shape[0], -1) for k in keys}

        # flatten of shape [bs] to [bs, L_q]
        q_batch = self._concat_questions_and_answers(batch, fields=["question", "answer"])
        pprint_batch(q_batch)
        # keys=["question.input_ids", "question.attention_mask"]
        # q_batch = {k: batch[k].view(batch[k].shape[0], -1) for k in keys}

        return {**q_batch, **d_batch}

    def update_metrics(self, output: Batch, split: Split) -> None:
        """update the metrics of the given split."""
        # TODO: ADD TARGETS
        loss = (output.get(k, None) for k in ("loss"))
        self.answer_metrics.update(split, loss)

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
