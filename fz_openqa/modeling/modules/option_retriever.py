from enum import Enum
from typing import Any
from typing import Optional

import einops
import rich
import torch
import torch.nn.functional as F
from datasets import Split
from torch import Tensor

from ...utils.pretty import pprint_batch
from .utils.gradients import batch_backprop_grads
from .utils.gradients import GradExpression
from .utils.gradients import in_batch_grads
from .utils.gradients import supervised_loss
from .utils.gradients import variational_grads
from .utils.utils import flatten_first_dims
from fz_openqa.modeling.modules.base import Module
from fz_openqa.utils.datastruct import Batch


class Similarity(Enum):
    CLS = "cls"
    COLBERT = "colbert"


class OptionRetriever(Module):
    """
    A model for multiple-choice OpenQA.
    This is a retriever-only model allowing both for retrieval and option selection.
    The model is described in : https://hackmd.io/tQ4_EDx5TMyQwwWO1rvUIA
    """

    _required_feature_names = []

    _required_eval_feature_names = [
        "question.input_ids",
        "question.attention_mask",
        "document.input_ids",
        "document.attention_mask",
        "answer.target",
    ]

    # prefix for the logged metrics
    task_id: Optional[str] = None

    # metrics to display in the progress bar
    pbar_metrics = [
        "train/reader/logp",
        "validation/reader/logp",
        "train/reader/Accuracy",
        "validation/reader/Accuracy",
    ]

    # require heads
    _required_heads = [
        "question_reader",
        "document_reader",
        "question_retriever",
        "document_retriever",
    ]

    def __init__(
        self,
        *args,
        alpha: float = 0,
        resample_k: Optional[int] = None,
        grad_expr: GradExpression = GradExpression.BATCH_BACKPROP,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.grad_expr = GradExpression(grad_expr)
        self.resample_k = resample_k
        head = next(iter(self.heads.values()))
        self.similarity = Similarity(head.id)
        self.alpha = alpha

    def _init_metrics(self, prefix: str):
        """Initialize the metrics for each split."""
        self.metrics = self._get_base_metrics(prefix=f"{prefix}reader/")

        self.retriever_metrics = self._get_base_metrics(
            prefix=f"{prefix}retriever/", topk=[None, 3, 5, 10, 20, 50, 100]
        )

    def _forward(self, batch: Batch, **kwargs) -> Batch:
        q_shape = d_shape = None
        output = {}
        if "document.input_ids" in batch:
            bs, n_opts, n_docs, _ = batch["document.input_ids"].shape
            # flatten the batch_size and n_options and n_docs dimensions
            d_batch = flatten_first_dims(
                batch,
                n_dims=3,
                keys=["document.input_ids", "document.attention_mask"],
            )

            # process the document with the backbone
            h_heads = self._backbone(
                d_batch,
                prefix="document",
                heads=["document_reader", "document_retriever"],
                **kwargs,
            )
            # reshape and return
            for k, v in h_heads.items():
                tag = k.split("_")[-1]
                v = einops.rearrange(
                    v,
                    "(bs n_opts n_docs) ... -> bs n_opts n_docs ...",
                    bs=bs,
                    n_opts=n_opts,
                    n_docs=n_docs,
                )
                output[f"_hd_{tag}_"] = v

        if "question.input_ids" in batch:
            bs, n_opts, _ = batch["question.input_ids"].shape
            # flatten the batch_size and n_options dimensions
            q_batch = flatten_first_dims(
                batch,
                n_dims=2,
                keys=["question.input_ids", "question.attention_mask"],
            )

            # process the document with the backbone
            q_heads = self._backbone(
                q_batch,
                prefix="question",
                heads=["question_reader", "question_retriever"],
                **kwargs,
            )

            # reshape and return
            for k, v in q_heads.items():
                tag = k.split("_")[-1]
                v = einops.rearrange(v, "(bs n_opts) ... -> bs n_opts ...", bs=bs, n_opts=n_opts)
                output[f"_hq_{tag}_"] = v

        pprint_batch(output, "forward", silent=True)

        if all(d is not None for d in (d_shape, q_shape)):
            if not d_shape[:2] == q_shape[:2]:
                raise ValueError(
                    f"Expected 2 first dimensions to be equal, "
                    f"got documents of shape: {d_shape} and "
                    f"questions of shape{q_shape}"
                )

        return output

    def _step(self, batch: Batch, **kwargs: Any) -> Batch:
        """
        Compute the forward pass for the question and the documents.
        """
        # check features, check that the first document of each question is positive
        # process the batch using BERT and the heads

        d_batch = {k: v for k, v in batch.items() if k.startswith("document.")}
        q_batch = {k: v for k, v in batch.items() if k.startswith("question.")}

        if self.resample_k:

            # compute documents logits
            with torch.no_grad():
                d_out = self._forward(d_batch, **kwargs)

            # compute questions logits
            output = self._forward(q_batch, **kwargs)

            # compute the score and sample k without replacement
            with torch.no_grad():
                retriever_score = self._compute_score(
                    hd=d_out["_hd_retriever_"], hq=output["_hq_retriever_"]
                )

                # sample k documents
                soft_samples = F.gumbel_softmax(retriever_score, hard=False, dim=-1)
                sample_ids = soft_samples.topk(self.resample_k, dim=-1)[1]

                # gather the sampled documents and update d_batch
                for k, v in d_batch.items():
                    if isinstance(v, torch.Tensor):
                        leaf_shape = v.shape[len(sample_ids.shape) :]
                        _index = sample_ids.view(*sample_ids.shape, *(1 for _ in leaf_shape))
                        _index = _index.expand(*sample_ids.shape, *leaf_shape)
                        d_batch[k] = v.gather(index=_index, dim=2)
        else:
            # compute questions logits
            output = self._forward(q_batch, **kwargs)

        # compute the document logits
        output.update(self._forward(d_batch, **kwargs))
        keys = [
            "_hq_reader_",
            "_hd_reader_",
            "_hq_retriever_",
            "_hd_retriever_",
        ]
        hq_reader, hd_reader, hq_retriever, hd_retriever = (output[k] for k in keys)
        # compute the score for each pair `f([q_j; a_j], d_jk)`
        reader_score = self._compute_score(hd=hd_reader, hq=hq_reader)
        retriever_score = self._compute_score(hd=hd_retriever, hq=hq_retriever)

        if self.grad_expr == GradExpression.BATCH_BACKPROP:
            step_output = batch_backprop_grads(
                reader_score=reader_score,
                retriever_score=retriever_score,
                targets=batch["answer.target"],
                grad_expr=self.grad_expr,
            )
        else:
            raise ValueError(f"Unknown grad_expr: {self.grad_expr}")

        # auxiliary loss
        supervised_loss_out = supervised_loss(retriever_score, batch["document.match_score"])
        supervised_loss_ = supervised_loss_out.get("retriever_loss", 0)
        if self.alpha > 0:
            step_output["loss"] += self.alpha * supervised_loss_
        step_output.update(supervised_loss_out)

        return step_output

    def _resample_k(self, batch: Batch, k: int) -> Batch:
        """Re-sample k documents from the batch."""
        pprint_batch(batch, "resample_k", silent=False)
        return batch

    def _compute_score(
        self,
        *,
        hd: Tensor,
        hq: Tensor,
        doc_ids: Optional[Tensor] = None,
        across_batch: bool = False,
    ) -> Tensor:
        """compute the score for each option and document $f([q;a], d)$ : [bs, n_options, n_docs]"""
        if not across_batch:
            if self.similarity == Similarity.CLS:
                return torch.einsum("boh, bodh -> bod", hq, hd)
            elif self.similarity == Similarity.COLBERT:
                scores = torch.einsum("bouh, bodvh -> boduv", hq, hd)
                max_scores, _ = scores.max(-1)
                return max_scores.sum(-1)
            else:
                raise ValueError(f"Unknown similarity: {self.similarity}, Similarity={Similarity}")
        else:
            if doc_ids is None:
                raise ValueError("doc_ids is required for non-element-wise computation")

            # get the unique list of documents vectors
            hd = einops.rearrange(hd, "bs opts docs ... -> (bs opts docs) ...")
            doc_ids = einops.rearrange(doc_ids, "bs opts docs -> (bs opts docs)")
            udoc_ids, uids = torch.unique(doc_ids, return_inverse=True)
            hd = hd[uids]
            if self.similarity == Similarity.CLS:
                return torch.einsum("boh, mh -> bom", hq, hd)
            elif self.similarity == Similarity.COLBERT:
                scores = torch.einsum("bouh, mvh -> bomuv", hq, hd)
                max_scores, _ = scores.max(-1)
                return max_scores.sum(-1)
            else:
                raise ValueError(f"Unknown similarity: {self.similarity}, Similarity={Similarity}")

    def _reduce_step_output(self, output: Batch) -> Batch:
        """
        Gather losses and logits from all devides and return
        """

        # average losses
        for k in ["loss", "reader/logp"]:
            y = output.get(k, None)
            if y is not None:
                output[k] = y.mean()

        return output

    def update_metrics(self, output: Batch, split: Split) -> None:
        """update the metrics of the given split."""
        logits, targets = (output.get(k, None) for k in ("_reader_logits_", "_reader_targets_"))
        self.metrics.update(split, logits, targets)

        retrieval_logits, retrieval_targets = (
            output.get(k, None) for k in ("_retriever_logits_", "_retriever_targets_")
        )
        if retrieval_logits is not None and retrieval_logits.numel() > 0:
            self.retriever_metrics.update(split, retrieval_logits, retrieval_targets)

    def reset_metrics(self, split: Optional[Split] = None) -> None:
        """
        Reset the metrics corresponding to `split` if provided, else
        reset all the metrics.
        """
        self.metrics.reset(split)
        self.retriever_metrics.reset(split)

    def compute_metrics(self, split: Optional[Split] = None) -> Batch:
        """
        Compute the metrics for the given `split` else compute the metrics for all splits.
        The metrics are return after computation.
        """
        return {
            **self.metrics.compute(split),
            **self.retriever_metrics.compute(split),
        }
