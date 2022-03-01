import warnings
from enum import Enum
from typing import Any
from typing import Optional

import einops
import numpy as np
import rich
import torch
import torch.nn.functional as F
from datasets import Split
from torch import Tensor

from ...utils.fingerprint import get_fingerprint
from ...utils.pretty import pprint_batch
from .metrics import SafeMetricCollection
from .metrics import SplitMetrics
from .utils.gradients import GradExpression
from .utils.gradients import InBatchGradients
from .utils.gradients import supervised_loss
from .utils.gradients import VariationalGradients
from .utils.total_epoch_metric import TotalEpochMetric
from .utils.utils import flatten_first_dims
from fz_openqa.modeling.modules.base import Module
from fz_openqa.utils.datastruct import Batch


class Similarity(Enum):
    DENSE = "dense"
    COLBERT = "colbert"


def hash_bert(bert):
    bert_params = {k: get_fingerprint(v) for k, v in bert.named_parameters() if "encoder." in k}
    bert_fingerprint = get_fingerprint(bert_params)
    is_training = bert.training
    bert.eval()
    state = np.random.RandomState(0)
    x = state.randint(0, bert.config.vocab_size - 1, size=(3, 512))
    x = torch.from_numpy(x)
    h = bert(x).last_hidden_state
    input_fingerprint = get_fingerprint(x)
    output_fingerprint = get_fingerprint(h)
    if is_training:
        bert.train()

    return {
        "bert_fingerprint": bert_fingerprint,
        "input_fingerprint": input_fingerprint,
        "output_fingerprint": output_fingerprint,
    }


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
        max_batch_size: Optional[int] = None,
        eval_topk: Optional[int] = 20,
        resample_k: Optional[int] = None,
        grad_expr: GradExpression = GradExpression.IN_BATCH,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.resample_k = resample_k
        self.max_batch_size = max_batch_size
        head = next(iter(self.heads.values()))
        self.similarity = Similarity(head.id)
        self.alpha = alpha

        # init the estimator
        grad_expr = GradExpression(grad_expr)
        estimator_cls = {
            GradExpression.IN_BATCH: InBatchGradients,
            GradExpression.VARIATIONAL: VariationalGradients,
        }[grad_expr]
        self.estimator = estimator_cls(eval_topk=eval_topk)

        for k, hs in hash_bert(self.bert).items():
            rich.print(f"> {k}={hs}")

    def _init_metrics(self, prefix: str):
        """Initialize the metrics for each split."""
        self.metrics = self._get_base_metrics(prefix=f"{prefix}reader/")

        self.total_logp_metrics = self._init_total_logp_metrics(prefix)

        self.retriever_metrics = self._get_base_metrics(
            prefix=f"{prefix}retriever/", topk=[None, 3, 5, 10, 20, 50, 100]
        )

    def _init_total_logp_metrics(self, prefix):
        metric_kwargs = {"compute_on_step": False, "dist_sync_on_step": True}
        metrics = SafeMetricCollection(
            {"reader/total-logp": TotalEpochMetric(**metric_kwargs)},
            prefix=prefix,
        )
        return SplitMetrics(metrics)

    def _forward(self, batch: Batch, **kwargs) -> Batch:
        output = {}

        if "document.input_ids" in batch:
            output.update(self._forward_field(batch, "document", **kwargs))

        if "question.input_ids" in batch:
            output.update(self._forward_field(batch, "question", **kwargs))

        return output

    def _forward_field(self, batch: Batch, field: str, silent: bool = True, **kwargs) -> Batch:
        original_shape = batch[f"{field}.input_ids"].shape
        pprint_batch(batch, f"forward {field}", silent=silent)
        if self.training:
            max_batch_size = None
        else:
            max_batch_size = self.max_batch_size

        # flatten the batch
        flat_batch = flatten_first_dims(
            batch,
            n_dims=len(original_shape) - 1,
            keys=[f"{field}.input_ids", f"{field}.attention_mask"],
        )
        # process the document with the backbone
        h_heads = self._backbone(
            flat_batch,
            prefix=f"{field}",
            heads=[f"{field}_reader", f"{field}_retriever"],
            max_batch_size=max_batch_size,
            **kwargs,
        )
        pprint_batch(h_heads, f"h_heads {field}", silent=silent)

        # reshape and return
        output = {}
        for k, v in h_heads.items():
            tag = k.split("_")[-1]  # reader / retriever
            v = v.view(*original_shape[:-1], *v.shape[1:])
            name = {"document": "hd", "question": "hq"}[field]
            output[f"_{name}_{tag}_"] = v

        pprint_batch(output, f"output {field}", silent=silent)

        return output

    def _step(self, batch: Batch, **kwargs: Any) -> Batch:
        """
        Compute the forward pass for the question and the documents.
        """
        # check features, check that the first document of each question is positive
        # process the batch using BERT and the heads

        d_batch = {k: v for k, v in batch.items() if k.startswith("document.")}
        q_batch = {k: v for k, v in batch.items() if k.startswith("question.")}
        n_docs = d_batch["document.input_ids"].shape[2]
        output = {}
        step_output = {}
        is_supervised_loss_computed = False
        if self.resample_k is not None and n_docs > self.resample_k:
            warnings.warn(f"Resampling documents from {n_docs} to {self.resample_k}")

            # compute documents logits
            with torch.no_grad():
                d_out = self._forward(d_batch, **kwargs)

            # compute questions logits
            output.update(self._forward(q_batch, **kwargs))

            # compute the score and sample k without replacement
            with torch.no_grad():
                retriever_score = self._compute_score(
                    hd=d_out["_hd_retriever_"], hq=output["_hq_retriever_"]
                )

                # log the retriever accuracy
                if "document.match_score" in d_batch.keys():
                    supervised_loss_out = supervised_loss(
                        retriever_score, d_batch["document.match_score"]
                    )
                    is_supervised_loss_computed = True
                    step_output.update(supervised_loss_out)

                # sample k documents
                soft_samples = F.gumbel_softmax(retriever_score, hard=False, dim=-1)
                sample_ids = soft_samples.topk(self.resample_k, dim=-1)[1]

                # re-sample the documents
                for k, v in d_batch.items():
                    if isinstance(v, torch.Tensor):
                        leaf_shape = v.shape[len(sample_ids.shape) :]
                        _index = sample_ids.view(*sample_ids.shape, *(1 for _ in leaf_shape))
                        _index = _index.expand(*sample_ids.shape, *leaf_shape)
                        d_batch[k] = v.gather(index=_index, dim=2)

        else:
            # compute questions logits
            output.update(self._forward(q_batch, **kwargs))

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

        # retriever diagnostics
        self._retriever_diagnostics(
            retriever_score,
            d_batch.get("document.retrieval_score", None),
            d_batch.get("document.retrieval_rank", None),
            output=step_output,
        )

        # compute the gradients
        step_output.update(
            self.estimator(
                reader_score=reader_score,
                retriever_score=retriever_score,
                targets=batch["answer.target"],
            )
        )

        # auxiliary loss
        if self.alpha > 0 or not is_supervised_loss_computed:
            if "document.match_score" in d_batch.keys():
                supervised_loss_out = supervised_loss(
                    retriever_score, d_batch["document.match_score"]
                )
                supervised_loss_ = supervised_loss_out.get("retriever/loss", 0)
                if not is_supervised_loss_computed:
                    step_output.update(supervised_loss_out)
                if self.alpha > 0:
                    warnings.warn(f"Using alpha={self.alpha}")
                    step_output["loss"] += self.alpha * supervised_loss_
        step_output["retriever/alpha"] = self.alpha

        return step_output

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
            if self.similarity == Similarity.DENSE:
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
            if self.similarity == Similarity.DENSE:
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
        for k, v in output.items():
            if not str(k).startswith("_") and not str(k).endswith("_"):
                if isinstance(v, torch.Tensor):
                    v = v.float().mean()
                output[k] = v

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

        if "reader/logp" in output:
            self.total_logp_metrics.update(split, output["reader/logp"], None)

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
            **self.total_logp_metrics.compute(split),
        }

    @staticmethod
    @torch.no_grad()
    def _retriever_diagnostics(
        retriever_score: Tensor,
        retrieval_scores: Optional[Tensor],
        retrieval_rank: Optional[Tensor],
        *,
        output,
    ):
        """
        Compute diagnostics for the rank of the retrieved documents.

        NB: `retriever_probs` corresponds to the probs of the trained model whereas
        `retrieval_*` correspond to the probs of the model used for indexing.
        """
        retriever_score = retriever_score.clone().detach()
        retriever_log_probs = retriever_score.log_softmax(dim=-1)
        retriever_probs = retriever_log_probs.exp()

        # arg_i(cdf=p)
        sorted_probs = retriever_probs.sort(dim=-1, descending=True).values
        cdf = sorted_probs.cumsum(dim=-1)
        for p in [0.5, 0.9]:
            arg_cdf_90 = 1 + (cdf - p).abs().argmin(dim=-1)
            output[f"retriever/arg_cdf_{int(100 * p)}"] = arg_cdf_90.float().mean()

        # entropy `H(p(D | q, A))`
        retriever_entropy = -(retriever_probs * retriever_log_probs).sum(dim=(1, 2))
        output["retriever/entropy"] = retriever_entropy.mean()

        if retrieval_scores is not None:
            #  truncate `retrieval_scores` to avoid `NaN` and compute `log r(D | q, A)`
            M = retrieval_scores.max(dim=-1, keepdim=True).values
            retrieval_scores = retrieval_scores - M
            retrieval_scores = retrieval_scores.clip(min=-1e6)
            retrieval_log_probs = retrieval_scores.log_softmax(dim=-1)

            # `KL( p(D|q, A) || r(D|q, A) )`
            kl_div = retriever_probs * (retriever_log_probs - retrieval_log_probs)
            kl_div = kl_div.sum(dim=(1, 2))
            output["retriever/kl_div"] = kl_div.mean()

        # retrieval rank info
        if retrieval_rank is not None:
            # retrieval rank weighted by the probability of the retrieved document
            weighted_rank = retriever_probs * retrieval_rank
            output["retriever/weighted_rank"] = weighted_rank.sum(-1).mean()

            # rank of the most likely document
            top_idx = retriever_probs.argmax(dim=-1).unsqueeze(-1)
            top_rank = retrieval_rank.gather(dim=-1, index=top_idx)
            output["retriever/top_rank"] = top_rank.float().mean()

            # min-max of the retrieval rank
            output["retriever/n_samples"] = retrieval_rank.size(-1)
            output["retriever/max_sampled_rank"] = (
                (retrieval_rank.max(dim=-1).values).float().mean()
            )
            output["retriever/min_sampled_rank"] = (
                (retrieval_rank.min(dim=-1).values).float().mean()
            )
