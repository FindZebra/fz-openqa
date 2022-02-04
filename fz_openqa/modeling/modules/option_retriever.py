from __future__ import annotations

import math
import string
import warnings
from enum import Enum
from typing import Any
from typing import Dict
from typing import Optional

import einops
import numpy as np
import rich
import torch
import torch.nn.functional as F
from datasets import Split
from torch import Tensor

from ...utils.pretty import pprint_batch
from .metrics import SafeMetricCollection
from .metrics import SplitMetrics
from .utils.gradients import GradExpression
from .utils.gradients import InBatchGradients
from .utils.gradients import ReinforceGradients
from .utils.gradients import supervised_loss
from .utils.gradients import VariationalGradients
from .utils.total_epoch_metric import TotalEpochMetric
from .utils.utils import flatten_first_dims
from fz_openqa.modeling.modules.base import Module
from fz_openqa.utils.datastruct import Batch


class Similarity(Enum):
    DENSE = "dense"
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
        max_batch_size: Optional[int] = None,
        eval_topk: Optional[int] = 20,
        resample_k: Optional[int] = None,
        grad_expr: GradExpression = GradExpression.IN_BATCH,
        head_map: None = None,
        share_heads: bool = True,
        temperature: float = 1.0,
        **kwargs,
    ):
        assert head_map is None, "`head_map` cannot be set manually for this model."
        if share_heads:
            head_map = {
                "question_reader": "reader",
                "document_reader": "reader",
                "question_retriever": "retriever",
                "document_retriever": "retriever",
            }

        super().__init__(*args, head_map=head_map, **kwargs)

        self.resample_k = resample_k
        self.max_batch_size = max_batch_size
        head = next(iter(self.heads.values()))
        self.similarity = Similarity(head.id)
        self.alpha = alpha
        self.temperature = temperature

        # init the estimator
        grad_expr = GradExpression(grad_expr)
        estimator_cls = {
            GradExpression.IN_BATCH: InBatchGradients,
            GradExpression.VARIATIONAL: VariationalGradients,
            GradExpression.REINFORCE: ReinforceGradients,
        }[grad_expr]
        self.estimator = estimator_cls(eval_topk=eval_topk)

        # init the QMASK gate parameter
        self.qmask_gate = torch.nn.Parameter(torch.tensor(0.0))

        # punctuation masking
        self.skiplist = [
            self.tokenizer.encode(symbol, add_special_tokens=False)[0]
            for symbol in string.punctuation
        ]

        # reader attention head
        # hdim = 256
        # bert_output_size = self.bert.config.hidden_size
        # self.reader_num_attention_heads = 8
        # self.reader_kv = torch.nn.Linear(bert_output_size, 2 * hdim)
        # self.reader_q = torch.nn.Linear(bert_output_size, hdim)
        # self.reader_residual = torch.nn.Linear(bert_output_size, hdim)
        # self.reader_attention_dropout = torch.nn.Dropout(
        #     self.bert.config.attention_probs_dropout_prob
        # )
        # self.reader_hidden_dropout = torch.nn.Dropout(self.bert.config.hidden_dropout_prob)
        # self.reader_norm = torch.nn.LayerNorm(hdim)
        # self.reader_projection = torch.nn.Linear(hdim, 3 * hdim)
        #
        # # init
        # self.reader_kv.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
        # self.reader_q.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
        # self.reader_residual.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
        # self.reader_projection.weight.data.normal_(mean=0.0,
        # std=self.bert.config.initializer_range)
        # self.reader_kv.bias.data.zero_()
        # self.reader_q.bias.data.zero_()
        # self.reader_residual.bias.data.zero_()
        # self.reader_projection.bias.data.zero_()
        # self.reader_norm.bias.data.zero_()
        # self.reader_norm.weight.data.fill_(1.0)

    def _compute_reader_scores(
        self,
        *,
        hd: Tensor,
        hq: Tensor,
        mask: Optional[Tensor] = None,
    ):
        # pprint_batch({"hd": hd, "hq": hq, "mask": mask}, "Soft reader scores")

        bs, n_opts, n_docs, doc_len, hdim = hd.shape
        bs, n_opts, q_len, hdim = hq.shape

        # compute the features for the attention layer
        q = self.reader_q(hq)
        q = einops.rearrange(
            q, "bs opts l (heads h) -> bs opts heads l h", heads=self.reader_num_attention_heads
        )
        kv = self.reader_kv(hd)
        kv = einops.rearrange(
            kv,
            "bs opts docs l (heads h) -> bs opts docs heads l h",
            heads=self.reader_num_attention_heads,
        )
        k, v = kv.chunk(2, dim=-1)

        # compute the cross-attention weights
        weights = torch.einsum("bohux, bodhvx -> bodhuv", q, k)
        weights = weights / math.sqrt(weights.shape[-1])
        weights = weights.softmax(dim=-1)
        weights = self.reader_attention_dropout(weights)

        # cross-attention out
        h_qd = torch.einsum("bodhuv, bodhvx -> bodhux", weights, v)
        h_qd = einops.rearrange(
            h_qd,
            "bs opts docs heads l h -> bs opts docs l (heads h)",
            heads=self.reader_num_attention_heads,
        )

        # apply dropout + layer norm + residual
        h_qd = self.reader_hidden_dropout(h_qd)
        skip = self.reader_residual(hq).unsqueeze(2)
        h_qd = self.reader_norm(h_qd + skip)

        # pprint_batch({"h_qd": h_qd}, "Soft reader scores : 2")

        # final self-attention layer
        qkv = self.reader_projection(h_qd)
        qkv = einops.rearrange(
            qkv,
            "bs opts docs l (heads h) -> bs opts docs heads l h",
            heads=self.reader_num_attention_heads,
        )
        q, k, v = qkv.chunk(3, dim=-1)
        q = q[..., 0, :]  # select the CLS token of the query

        # compute the self-attention weights
        weights = torch.einsum("bodhx, bodhvx -> bodhv", q, k)
        weights = weights / math.sqrt(weights.shape[-1])
        if mask is not None:
            mask_ = mask.bool()
            mask_ = mask_.view(*mask_.shape[:2], 1, 1, mask_.shape[-1])
            weights = weights.masked_fill(~mask_, -float("inf"))
        weights = weights.softmax(dim=-1)
        weights = self.reader_attention_dropout(weights)

        # compute the final score of shape [bs, n_opts, n_docs]
        score = torch.einsum("bodhv, bodhvx -> bod", weights, v)

        return score

    def _init_metrics(self, prefix: str):
        """Initialize the metrics for each split."""
        self.reader_metrics = self._get_base_metrics(prefix=f"{prefix}reader/")
        self.retriever_reading_metrics = self._get_base_metrics(prefix=f"{prefix}reader/retriever-")

        self.retriever_metrics = self._get_base_metrics(
            prefix=f"{prefix}retriever/", topk=[None, 3, 5, 10, 20, 50, 100]
        )

        # self.total_logp_metrics = self._init_total_logp_metrics(prefix)

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

    def _forward_field(
        self,
        batch: Batch,
        field: str,
        silent: bool = True,
        max_batch_size: Optional[int] = None,
        **kwargs,
    ) -> Batch:
        original_shape = batch[f"{field}.input_ids"].shape
        pprint_batch(batch, f"forward {field}", silent=silent)

        if max_batch_size is None:
            max_batch_size = self.max_batch_size
        elif max_batch_size < 0:
            max_batch_size = None

        # flatten the batch
        flat_batch = flatten_first_dims(
            batch,
            n_dims=len(original_shape) - 1,
            keys=[f"{field}.input_ids", f"{field}.attention_mask"],
        )

        # compute question output mask
        if field == "question":
            # inputs_ids = flat_batch["question.input_ids"]
            mask = torch.ones_like(flat_batch["question.attention_mask"].float())
            mask[mask == self._pad_token_id] = 0
            # qmask_id = self.spec_tokens[QUERY_MASK]
            # mask = torch.where(inputs_ids == qmask_id, self.qmask_gate, mask)
        elif field == "document":
            inputs_ids = flat_batch["document.input_ids"]
            mask = torch.ones_like(flat_batch["document.attention_mask"].float())
            mask[mask == self._pad_token_id] = 0
            for w in self.skiplist:
                mask[inputs_ids == w] = 0
        else:
            mask = None

        # process the document with the backbone
        h_heads = self._backbone(
            flat_batch,
            prefix=f"{field}",
            heads=[f"{field}_reader", f"{field}_retriever"],
            max_batch_size=max_batch_size,
            mask=mask,
            **kwargs,
        )
        if "last_hidden_state" in h_heads.keys():
            h_heads[f"{field}_last_hidden_state"] = h_heads.pop("last_hidden_state")
        pprint_batch(h_heads, f"h_heads {field}", silent=silent)

        # reshape and return
        output = {}
        for k, v in h_heads.items():
            _, *tag = k.split("_")  # reader / retriever
            tag = "_".join(tag)
            v = v.view(*original_shape[:-1], *v.shape[1:])
            name = {"document": "hd", "question": "hq"}[field]
            output[f"_{name}_{tag}_"] = v

        pprint_batch(output, f"output {field}", silent=silent)

        return output

    def _step(self, batch: Batch, silent=True, **kwargs: Any) -> Batch:
        """
        Compute the forward pass for the question and the documents.
        """
        # check features, check that the first document of each question is positive
        # process the batch using BERT and the heads

        d_batch = {k: v for k, v in batch.items() if k.startswith("document.")}
        q_batch = {k: v for k, v in batch.items() if k.startswith("question.")}
        output = {}
        step_output = {}
        is_supervised_loss_computed = False
        # `max_batch_size` is used to limit the number of samples in the batch, it is
        # only used during eval, except when resampling..
        max_batch_size_eval = -1 if self.training else self.max_batch_size

        # Register the document and question shape
        # if the documents are of shape [batch_size, n_docs, seq_len], documents
        # will be expanded to shape [batch_size, n_options, seq_len].
        doc_shape = d_batch["document.input_ids"].shape[:-1]
        query_shape = q_batch["question.input_ids"].shape[:-1]
        if len(doc_shape) != 3:
            assert len(doc_shape) == 2
            doc_target_shape = (doc_shape[0], query_shape[1], doc_shape[1])
        else:
            doc_target_shape = None

        pprint_batch(d_batch, f"Option retriever::d_batch::in: {doc_target_shape}", silent=silent)
        pprint_batch(q_batch, "Option retriever::q_batch::in", silent=silent)

        if self.resample_k is not None:
            pprint_batch(d_batch, "d_batch", silent=silent)

            # compute documents logits, and potentially expand to match
            # the shape [batch_size, n_options, n_documents, ...]
            with torch.no_grad():
                d_out = self._forward(
                    d_batch, silent=silent, max_batch_size=self.max_batch_size, **kwargs
                )
                d_out = {k: self._expand_to_shape(v, doc_target_shape) for k, v in d_out.items()}

            # compute questions logits, shape [batch_size, n_options, ...]
            output.update(
                self._forward(q_batch, silent=silent, max_batch_size=max_batch_size_eval, **kwargs)
            )
            pprint_batch({**output, **d_out}, "Option retriever::resampling::input", silent=silent)

            # compute the score `log p(d |q, a)`and sample `k` documents without replacement
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
                original_retrieval_score = d_batch.get("document.retrieval_score", None)
                retriever_score = self._mask_scores(retriever_score, original_retrieval_score)
                soft_samples = F.gumbel_softmax(
                    retriever_score / self.temperature, hard=False, dim=-1
                )
                sample_ids = soft_samples.topk(self.resample_k, dim=-1)[1]

                # re-sample the documents
                for k, v in d_batch.items():
                    v = self._expand_to_shape(v, doc_target_shape)
                    v = self._select_with_index(v, sample_ids)
                    if isinstance(v, torch.Tensor):
                        v = v.contiguous()
                    d_batch[k] = v

                pprint_batch(d_batch, "Option retriever::resampling::output", silent=silent)
        else:
            # compute the questions logits
            output.update(
                self._forward(q_batch, max_batch_size=max_batch_size_eval, silent=silent, **kwargs)
            )

        # compute the document logits
        d_out = self._forward(d_batch, max_batch_size=max_batch_size_eval, silent=silent, **kwargs)
        output.update(d_out)
        pprint_batch(output, "Option retriever::outputs::final", silent=silent)

        keys = [
            "_hq_reader_",
            "_hd_reader_",
            "_hq_retriever_",
            "_hd_retriever_",
        ]
        hq_reader, hd_reader, hq_retriever, hd_retriever = (output[k] for k in keys)
        # compute the score for each pair `f([q_j; a_j], d_jk)`
        # reader_score = self._compute_reader_scores(hd=output["_hd_last_hidden_state_"],
        #                                            hq=output["_hq_last_hidden_state_"],
        #                                            mask=q_batch["question.attention_mask"])
        reader_score = self._compute_score(
            hd=hd_reader, hq=hq_reader, mask=q_batch["question.attention_mask"]
        )
        retriever_score = self._compute_score(
            hd=hd_retriever, hq=hq_retriever, mask=q_batch["question.attention_mask"]
        )

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
                retrieval_score=d_batch.get("document.retrieval_score", None),
                retrieval_log_prob=d_batch.get("document.retrieval_log_prob", None),
                retrieval_log_Z=d_batch.get("document.retrieval_log_Z", None),
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

    @staticmethod
    @torch.no_grad()
    def _retriever_diagnostics(
        retriever_score: Tensor,
        retrieval_scores: Optional[Tensor],
        retrieval_rank: Optional[Tensor],
        *,
        output: Dict,
    ):
        """
        Compute diagnostics for the rank of the retrieved documents.

        NB: `retriever_probs` corresponds to the probs of the trained model whereas
        `retrieval_*` correspond to the probs of the model used for indexing.
        """

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

    @staticmethod
    def _mask_scores(retriever_score: Tensor, original_retrieval_score: Optional[Tensor]) -> Tensor:
        if original_retrieval_score is not None:
            retrieval_score = OptionRetriever._expand_to_shape(
                original_retrieval_score, retriever_score.shape
            )
            # consider thant `retrieval_score` falling bellow this threshold are
            # are documents added as padding (see `SearchResult`)
            retriever_score[retrieval_score < -1e15] = -float("inf")

        return retriever_score

    @staticmethod
    def _select_with_index(v: Any | Tensor, index: Tensor) -> Any | Tensor:
        if not isinstance(v, torch.Tensor):
            return v
        leaf_shape = v.shape[len(index.shape) :]
        _index = index.view(*index.shape, *(1 for _ in leaf_shape))
        _index = _index.expand(*index.shape, *leaf_shape)
        v = v.gather(index=_index, dim=2)
        return v

    @staticmethod
    def _expand_to_shape(x: Tensor, doc_target_shape: Optional[torch.Size]) -> Tensor:
        if doc_target_shape is None:
            return x
        elif x.shape[: len(doc_target_shape)] != doc_target_shape:
            x = x.unsqueeze(1)
            x = x.expand(*doc_target_shape[:2], *x.shape[2:])
            return x
        else:
            return x

    def _compute_score(
        self,
        *,
        hd: Tensor,
        hq: Tensor,
        doc_ids: Optional[Tensor] = None,
        across_batch: bool = False,
        mask: Optional[Tensor] = None,
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
        Gather losses and logits from all devices and return
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
        reader_logits = output.get("_reader_logits_", None)
        reader_targets = output.get("_reader_targets_", None)
        retriever_reading_logits = output.get("_retriever_reading_logits_", None)
        self.reader_metrics.update(split, reader_logits, reader_targets)
        self.retriever_reading_metrics.update(split, retriever_reading_logits, reader_targets)

        retrieval_logits, retrieval_targets = (
            output.get(k, None) for k in ("_retriever_logits_", "_retriever_targets_")
        )
        if retrieval_logits is not None and retrieval_logits.numel() > 0:
            self.retriever_metrics.update(split, retrieval_logits, retrieval_targets)

        # if "reader/logp" in output:
        #     self.total_logp_metrics.update(split, output["reader/logp"], None)

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
            # **self.total_logp_metrics.compute(split),
        }
