import warnings
from dataclasses import dataclass
from typing import Any
from typing import List
from typing import Optional

import torch
from datasets import Split
from torch import Tensor
from torch.nn import functional as F
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy

from ..functional import flatten
from ..functional import padless_cat
from .base import Model
from .metrics import SplitMetrics
from .utils import check_only_first_doc_positive
from .utils import expand_and_flatten
from .utils import flatten_first_dims
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.functional import batch_reduce


class ReaderMultipleChoice(Model):
    # name of the features required for a forward pass
    _required_feature_names = [
        "answer.target",
        "question.input_ids",
        "question.attention_mask",
        "document.input_ids",
        "document.attention_mask",
        "document.is_positive",
        "answer.input_ids",
        "answer.attention_mask",
    ]

    # named of the features required for evaluation
    _required_eval_feature_names = [
        "answer.target",
    ]

    # prefix for the logged metrics
    task_id: Optional[str] = "reading"

    # metrics to display
    pbar_metrics = [
        "train/reading/loss",
        "train/reading/Accuracy",
        "validation/reading/Accuracy",
        "train/reading/relevance-Accuracy",
        "validation/reading/relevance-Accuracy",
    ]

    def _init_metrics(self, prefix: str = ""):
        """Initialize a Metric for each split=train/validation/test
        fir both the answering model and the selection model"""
        metric_kwargs = {"compute_on_step": False, "dist_sync_on_step": True}

        def init_answer_metric():
            return MetricCollection([Accuracy(**metric_kwargs)], prefix=prefix)

        self.answer_metrics = SplitMetrics(init_answer_metric)

        def init_relevance_metric():
            return MetricCollection(
                [Accuracy(**metric_kwargs)], prefix=f"{prefix}relevance-"
            )

        self.relevance_metrics = SplitMetrics(init_relevance_metric)

    def _forward(self, batch: Batch, reshape: bool = True, **kwargs) -> Batch:
        """
        Compute the multiple choice answer model:
         * log p(a_i | q, e)
        where:
        *  log p(a_i| q, e) = Sim(BERT_CLS([q;d]), BERT_CLS(a_i))

        Future work:
        This is a very simple model, it will be improved in the next iterations.
        We can improve it either using a full interaction model
        * BERT_CLS([q;d;a_i])
        which requires roughly `N_q` times more GPU memory
        Or using a local interaction model `Sim` much similar to ColBERT.

        Input data:
        batch = {
        'document.input_ids': [batch_size, n_docs, L_d]
        'question.input_ids': [batch_size, L_q]
        'answer.input_ids': [batch_size, n_options, L_q]
        }
        """
        # todo batch = copy(batch)  # make sure the original batch is not modified
        # checks inputs, set parameters and concat the questions with the documents
        bs, n_options, *_ = batch["answer.input_ids"].shape
        b_, n_docs, *_ = batch["document.input_ids"].shape

        batch = self._expand_and_flatten_qd(batch, n_docs)

        # concatenate questions and documents such that there is no padding between Q and D
        qd_batch = self.concat_fields_across_dim_one(
            batch, ["question", "document"]
        )

        # compute contextualized representations
        heq = self.backbone(
            qd_batch["input_ids"][:, : self.max_length],
            attention_mask=qd_batch["attention_mask"][:, : self.max_length],
        )  # [bs*n_doc, h]

        ha = self.backbone(
            flatten(batch["answer.input_ids"]),
            flatten(batch["answer.attention_mask"]),
        )  # [bs * N_a, h]

        # infer the number of documents
        n_docs = heq.shape[0] // bs
        assert n_docs > 0, (
            f"number of documents should be at least one: "
            f"got n_docs={n_docs}, heq.shape={heq.shape}, bs={bs}"
        )

        # relevance model (see Dense Passage Retrieval)
        h_relevance = self.relevance_head(self.dropout(heq)).mean(-1)
        h_relevance = h_relevance.view(bs, n_docs)

        # answer-question final representation
        heq = self.qd_head(self.dropout(heq))  # [bs * n_doc, h]
        ha = self.a_head(self.dropout(ha)).view(
            bs, n_options, self.hparams.hidden_size
        )
        # dot-product model S(qd, a)
        heq = heq.view(bs, n_docs, *heq.shape[1:])
        S_qda = torch.einsum("bdh, bah -> bda", heq, ha)

        return {"_answer_logits_": S_qda, "_relevance_logits_": h_relevance}

    def _step(self, batch: Batch, **kwargs: Any) -> Batch:
        check_only_first_doc_positive(batch)
        relevance_targets = batch["document.is_positive"].long().argmax(-1)

        # arguments
        bs, n_docs = batch["document.input_ids"].shape[:2]

        # forward pass through the reader model
        output = self._forward(batch, **kwargs)

        # relevance loss
        # It is assumed that there is only a single positive document
        relevance_loss = self.batched_cross_entropy(
            relevance_targets, output["_relevance_logits_"]
        )

        # select the logits of the answering model corresponding to the positive document
        _index = relevance_targets.view(bs, 1, 1).expand(
            bs, 1, output["_relevance_logits_"].shape[-1]
        )
        answer_logits = (
            output["_answer_logits_"].gather(dim=1, index=_index).squeeze(1)
        )

        # compute the reader loss
        answer_targets: Tensor = batch["answer.target"]
        answer_loss = self.batched_cross_entropy(answer_targets, answer_logits)

        # final loss
        loss = answer_loss + relevance_loss

        return {
            "loss": loss,
            "relevance_loss": relevance_loss.detach(),
            "answer_loss": answer_loss.detach(),
            "_answer_targets_": answer_targets.detach(),
            "_relevance_targets_": relevance_targets.detach(),
            **{k: v.detach() for k, v in output.items()},
        }

    def batched_cross_entropy(self, targets, logits):
        """Compute cross entropy for each batch elements"""
        loss = F.cross_entropy(logits, targets, reduction="none")
        # keep one loss term per batch element: shape [batch_size, ]
        return batch_reduce(loss, torch.mean)

    def _reduce_step_output(self, output: Batch) -> Any:
        """
        Gather losses and logits from all devides and return
        """

        # average losses
        for k in ["loss", "relevance_loss", "answer_loss"]:
            y = output.get(k, None)
            if y is not None:
                output[k] = y.mean()

        return output

    def concat_fields_across_dim_one(self, batch: Batch, fields: List[str]):
        """
        Concatenate fields across the time dimension, and without padding
        """
        padded_batch = padless_cat(
            [
                {
                    "input_ids": batch[f"{key}.input_ids"],
                    "attention_mask": batch[f"{key}.attention_mask"],
                }
                for key in fields
            ],
            master_key="input_ids",
            pad_token=self._pad_token_id,
            aux_pad_tokens={"attention_mask": 0},
        )
        if len(padded_batch["input_ids"]) > self.max_length:
            warnings.warn("the tensor [question; document] was truncated.")

        return padded_batch

    def _expand_and_flatten_qd(self, batch: Batch, n_docs: int) -> Batch:
        # flatten documents of shape [bs, n_docs, L_docs] to [bs*n_docs, L_docs]
        batch.update(
            **flatten_first_dims(
                batch,
                2,
                keys=["document.input_ids", "document.attention_mask"],
            )
        )
        # expand questions to shape [bs, n_docs, L_1] and flatten to shape [bs*n_docs, L_q]
        batch.update(
            **expand_and_flatten(
                batch,
                n_docs,
                keys=["question.input_ids", "question.attention_mask"],
            )
        )
        return batch

    @staticmethod
    def expand_and_flatten(x: Tensor, n: int) -> Tensor:
        """
        Expand a tensor of shape [bs, *dims] as
        [bs, n, *dims] and flatten to [bs * n, *dims]
        """
        bs, *dims = x.shape
        x = x[:, None].expand(bs, n, *dims)
        x = x.contiguous().view(bs * n, *dims)
        return x

    def update_metrics(self, output: Batch, split: Split) -> None:
        """update the metrics of the given split."""
        answer_logits, answer_targets = (
            output.get(k, None)
            for k in ("_answer_logits_", "_answer_targets_")
        )
        self.answer_metrics.update(split, answer_logits, answer_targets)

        relevance_logits, relevance_targets = (
            output.get(k, None)
            for k in ("_relevance_logits_", "_relevance_targets_")
        )
        if relevance_targets is not None:
            self.relevance_metrics.update(
                split, relevance_logits, relevance_targets
            )

    def reset_metrics(self, split: Optional[Split] = None) -> None:
        """
        Reset the metrics corresponding to `split` if provided, else
        reset all the metrics.
        """
        self.answer_metrics.reset(split)
        self.relevance_metrics.reset(split)

    def compute_metrics(self, split: Optional[Split] = None) -> Batch:
        """
        Compute the metrics for the given `split` else compute the metrics for all splits.
        The metrics are return after computation.
        """
        return {
            **self.answer_metrics.compute(split),
            **self.relevance_metrics.compute(split),
        }
