import warnings
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
from .base import Module
from .metrics import SplitMetrics
from .utils import check_only_first_doc_positive
from .utils import expand_and_flatten
from .utils import flatten_first_dims
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.functional import batch_reduce


class ReaderMultipleChoice(Module):
    # name of the features required for a forward pass
    _required_feature_names = [
        "answer.target",
        "question.input_ids",
        "question.attention_mask",
        "document.input_ids",
        "document.attention_mask",
        "answer.input_ids",
        "answer.attention_mask",
    ]

    # named of the features required for evaluation
    _required_eval_feature_names = ["answer.target", "document.match_score"]

    # prefix for the logged metrics
    task_id: Optional[str] = "reader"

    # metrics to display
    pbar_metrics = [
        "train/reader/Accuracy",
        # "validation/reader/Accuracy",
        "train/reader/relevance-Accuracy",
        # "validation/reader/relevance-Accuracy",
    ]

    _required_heads = ["option", "evidence", "relevance"]

    def _init_metrics(self, prefix: str = ""):
        """Initialize a Metric for each split=train/validation/test
        fir both the answering model and the selection model"""
        metric_kwargs = {"compute_on_step": False, "dist_sync_on_step": True}

        def init_answer_metric():
            return MetricCollection([Accuracy(**metric_kwargs)], prefix=prefix)

        self.answer_metrics = SplitMetrics(init_answer_metric)

        def init_relevance_metric():
            return MetricCollection([Accuracy(**metric_kwargs)], prefix=f"{prefix}relevance-")

        self.relevance_metrics = SplitMetrics(init_relevance_metric)

    def _forward(self, batch: Batch, **kwargs) -> Batch:
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
        # checks inputs, set parameters and concat the questions with the documents
        bs, n_options, *_ = batch["answer.input_ids"].shape
        _, n_docs, *_ = batch["document.input_ids"].shape

        # concatenate questions and documents such that there is no padding between Q and D
        # todo: check switching q and d
        qd_batch = self._concat_questions_and_answers(batch, fields=["question", "document"])

        # compute evidence and relevance heads: [bs*n_doc, h]
        heq_heads = self._backbone(qd_batch, heads=["evidence", "relevance"])

        # compute answer head:  [bs * N_a, h]
        ha = self._backbone(
            {
                "input_ids": flatten(batch["answer.input_ids"]),
                "attention_mask": flatten(batch["answer.attention_mask"]),
            },
            head="option",
        )

        # get all the hidden representations
        heq = heq_heads["evidence"]  # [bs * n_doc, h]
        h_relevance = heq_heads["relevance"]
        heq = heq.view(bs, n_docs, *heq.shape[1:])
        h_relevance = h_relevance.view(bs, n_docs, *h_relevance.shape[1:])
        ha = ha.view(bs, n_options, heq.shape[-1])

        # relevance model (see Dense Passage Retrieval)
        # modified so it includes the similarity with the answers as well
        # todo: modify so each answer can be weighted differently,
        #  this is too much biased toward the answer
        #  edit: use the max instead of avg.
        S_relevance, _ = torch.einsum("bdh, bah -> bda", h_relevance, ha).max(-1)

        # answer-question final representation
        # dot-product model S(qd, a)
        S_qda = torch.einsum("bdh, bah -> bda", heq, ha)

        return {"_answer_logits_": S_qda, "_relevance_logits_": S_relevance}

    def _step(self, batch: Batch, **kwargs: Any) -> Batch:
        check_only_first_doc_positive(batch)

        # arguments
        bs, n_docs = batch["document.input_ids"].shape[:2]

        # forward pass through the reader model
        output = self._forward(batch, **kwargs)

        # relevance loss: match the positive document
        # It is assumed that there is only a single positive document
        relevance_targets = batch["document.match_score"].argmax(-1)
        relevance_loss = self._batched_cross_entropy(
            targets=relevance_targets, logits=output["_relevance_logits_"]
        )

        # select the logits of the answering model corresponding
        # to the positive document (relevance_target)
        _index = relevance_targets.view(bs, 1, 1).clone()
        _index = _index.expand(bs, 1, *output["_answer_logits_"].shape[2:])
        answer_logits = output["_answer_logits_"]  # [bs, n_documents, n_options, hdim]
        answer_logits = answer_logits.gather(dim=1, index=_index).squeeze(1)

        # compute the reader loss
        answer_targets: Tensor = batch["answer.target"]
        answer_loss = self._batched_cross_entropy(targets=answer_targets, logits=answer_logits)

        # final loss
        loss = answer_loss + relevance_loss

        return {
            "loss": loss,
            "relevance_loss": relevance_loss.detach(),
            "answer_loss": answer_loss.detach(),
            "_answer_targets_": answer_targets.detach(),
            "_relevance_targets_": relevance_targets.detach(),
            "_answer_logits_": answer_logits.detach(),
            "_relevance_logits_": output["_relevance_logits_"].detach(),
        }

    def _batched_cross_entropy(self, *, targets, logits):
        """Compute cross entropy for each batch elements"""
        loss = F.cross_entropy(logits, targets, reduction="none")
        # keep one loss term per batch element: shape [batch_size, ]
        return batch_reduce(loss, torch.mean)

    def _reduce_step_output(self, output: Batch) -> Batch:
        """
        Gather losses and logits from all devides and return
        """

        # average losses
        for k in ["loss", "relevance_loss", "answer_loss"]:
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
            "document",
        }, "Question and document fields must be provided."
        batch_size, n_docs = batch["document.input_ids"].shape[:2]

        # expand and flatten documents and questions such that
        # both are of shape [batch_size * n_docs, ...]
        batch = self._expand_and_flatten_qd(batch, n_docs)

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

    def _expand_and_flatten_qd(self, batch: Batch, n_docs: int) -> Batch:
        # flatten documents of shape [bs, n_docs, L_docs] to [bs*n_docs, L_docs]
        d_batch = flatten_first_dims(
            batch,
            2,
            keys=["document.input_ids", "document.attention_mask"],
        )
        # expand questions to shape [bs, n_docs, L_1] and flatten to shape [bs*n_docs, L_q]
        q_batch = expand_and_flatten(
            batch,
            n_docs,
            keys=["question.input_ids", "question.attention_mask"],
        )
        return {**q_batch, **d_batch}

    def update_metrics(self, output: Batch, split: Split) -> None:
        """update the metrics of the given split."""
        answer_logits, answer_targets = (
            output.get(k, None) for k in ("_answer_logits_", "_answer_targets_")
        )
        self.answer_metrics.update(split, answer_logits, answer_targets)

        relevance_logits, relevance_targets = (
            output.get(k, None) for k in ("_relevance_logits_", "_relevance_targets_")
        )
        if relevance_targets is not None:
            self.relevance_metrics.update(split, relevance_logits, relevance_targets)

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
