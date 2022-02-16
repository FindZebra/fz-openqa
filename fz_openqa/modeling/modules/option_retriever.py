from __future__ import annotations

import string
from copy import deepcopy
from enum import Enum
from typing import Any
from typing import Dict
from typing import Optional

import torch
import torch.nn.functional as F
from datasets import Split
from omegaconf import DictConfig
from torch import Tensor

from ...utils import maybe_instantiate
from ...utils.pretty import pprint_batch
from ..gradients import Gradients
from ..gradients import InBatchGradients
from ..heads.base import Head
from .utils.total_epoch_metric import TotalEpochMetric
from .utils.utils import flatten_first_dims
from fz_openqa.modeling.modules.base import Module
from fz_openqa.modeling.modules.utils.metrics import SafeMetricCollection
from fz_openqa.modeling.modules.utils.metrics import SplitMetrics
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
        reader_head: Head | DictConfig,
        retriever_head: Head | DictConfig,
        alpha: float = 0,
        resample_k: int = None,
        max_batch_size: Optional[int] = None,
        gradients: Gradients | DictConfig = InBatchGradients(),
        temperature: float = 1.0,
        use_gate: bool = False,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)

        # register the heads
        self.reader_head = maybe_instantiate(reader_head)
        self.retriever_head = maybe_instantiate(retriever_head)

        # register gates
        if use_gate:
            gate = torch.nn.Parameter(torch.zeros((1,)))
        else:
            gate = 1
        self.reader_gate = deepcopy(gate)
        self.retriever_gate = deepcopy(gate)

        # parameters
        self.resample_k = resample_k
        self.max_batch_size = max_batch_size
        self.alpha = alpha
        self.temperature = temperature

        # init the estimator
        self.estimator = maybe_instantiate(gradients)

        # punctuation masking
        self.skiplist = [
            self.tokenizer.encode(symbol, add_special_tokens=False)[0]
            for symbol in string.punctuation
        ]

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

    def mask(self, batch: Batch, field: str) -> Tensor:
        """prepare the head mask for the given field."""
        if field == "question":
            mask = torch.ones_like(batch["question.attention_mask"].float())
            mask[mask == self._pad_token_id] = 0
        elif field == "document":
            inputs_ids = batch["document.input_ids"]
            mask = torch.ones_like(batch["document.attention_mask"].float())
            mask[mask == self._pad_token_id] = 0
            for w in self.skiplist:
                mask[inputs_ids == w] = 0
        else:
            ValueError(f"Unknown field {field}")

        return mask

    def _forward(self, batch: Batch, predict: bool = True, **kwargs) -> Batch:
        output = {}

        if "document.input_ids" in batch:
            h = self._forward_field(batch, "document", **kwargs)
            if predict:
                mask = self.mask(batch, "document")
                h = self.retriever_head.preprocess(h, "document", mask=mask)
            output["_hd_"] = h

        if "question.input_ids" in batch:
            h = self._forward_field(batch, "question", **kwargs)
            if predict:
                mask = self.mask(batch, "question")
                h = self.retriever_head.preprocess(h, "question", mask=mask)
            output["_hq_"] = h

        return output

    def _forward_field(
        self,
        batch: Batch,
        field: str,
        silent: bool = True,
        max_batch_size: Optional[int] = None,
        **kwargs,
    ) -> Tensor:
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

        # process the document with the backbone
        h = self._backbone(
            flat_batch,
            prefix=f"{field}",
            max_batch_size=max_batch_size,
            **kwargs,
        )

        # reshape and return
        h = h.view(*original_shape[:-1], *h.shape[1:])
        return h

    def _step(self, batch: Batch, silent=True, **kwargs: Any) -> Batch:
        """
        Compute the forward pass for the question and the documents.
        """
        # check features, check that the first document of each question is positive
        # process the batch using BERT and the heads
        kwargs["predict"] = False
        d_batch = {k: v for k, v in batch.items() if k.startswith("document.")}
        q_batch = {k: v for k, v in batch.items() if k.startswith("question.")}
        output = {}
        step_output = {}
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
                # todo: resample using priority sampling
                retriever_score = self.retriever_gate * self.retriever_head(
                    hd=d_out["_hd_"],
                    hq=output["_hq_"],
                    q_mask=self.mask(batch, "question"),
                    d_mask=self.mask(batch, "document"),
                )

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

        reader_score = self.reader_gate * self.reader_head(
            hd=output["_hd_"],
            hq=output["_hq_"],
            q_mask=self.mask(batch, "question"),
            d_mask=self.mask(batch, "document"),
        )
        retriever_score = self.retriever_gate * self.retriever_head(
            hd=output["_hd_"],
            hq=output["_hq_"],
            q_mask=self.mask(batch, "question"),
            d_mask=self.mask(batch, "document"),
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
                retrieval_log_weight=d_batch.get("document.retrieval_log_weight", None),
            )
        )

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

    def step_end(
        self,
        output: Batch,
        split: Optional[Split],
        update_metrics: bool = True,
        filter_features: bool = True,
    ) -> Any:
        if split is not None and not split == Split.TRAIN:
            filter_features = False

        return super().step_end(
            output, split, filter_features=filter_features, update_metrics=update_metrics
        )
