from typing import Any

import torch
from datasets import Split
from torch import nn
from torch.nn import functional as F
from torchmetrics.classification import Accuracy

from .utils import check_first_doc_positive
from .utils import flatten_first_dims
from fz_openqa.modeling.evaluators.base import BaseEvaluator
from fz_openqa.modeling.evaluators.metrics import SafeMetricCollection
from fz_openqa.modeling.evaluators.metrics import SplitMetrics
from fz_openqa.modeling.similarities import Similarity
from fz_openqa.utils.datastruct import Batch


class InformationRetrievalGoldSupervised(BaseEvaluator):
    """
    Evaluates the Reader model `p(a_i | q, e, A)` using maximum likelihood estimation
    in a multiple choice QA context (A = [a_1,...a_P]). The loss is defined as:

        L =  sum_p log p(a_p | q, e, A) 1(p = a)

    where a is the index of the true answer.
    """

    _required_eval_feature_names = [
        "question.idx",
        "question.input_ids",
        "question.attention_mask",
        "document.input_ids",
        "document.attention_mask",
        "document.rank",
        "document.is_positive",
    ]

    def __init__(self, similarity: Similarity, **kwargs):
        super().__init__(**kwargs)
        self.similarity = similarity

    def init_metrics(self, prefix: str):
        """Initialize the metrics for each split."""
        metric_kwargs = {"compute_on_step": False, "dist_sync_on_step": True}

        def _name(k):
            return f"top{k}_Accuracy" if k is not None else "Accuracy"

        def init_metric():
            """generate a collection of topk accuracies."""
            return SafeMetricCollection(
                {
                    _name(k): Accuracy(top_k=k, **metric_kwargs)
                    for k in [None, 5, 10, 20, 50, 100]
                },
                prefix=prefix,
            )

        self.metrics = SplitMetrics(init_metric)

    def forward(
        self, model: nn.Module, batch: Batch, split: Split, **kwargs: Any
    ) -> Batch:
        """
        Compute the forward pass for the question and the documents.

        The input data is assumed to be of shape:
        batch = {
        'question.input_ids': [batch_size, L_q],
        'document.input_ids': [batch_size, n_docs, L_q]
        """
        # check features, check that the first document of each question is positive
        # and flatten the documents
        self.check_batch_type(batch)
        self.check_feature_names(batch)
        check_first_doc_positive(batch)
        # flatten documents inputs
        doc_batch = flatten_first_dims(
            batch,
            n_dims=2,
            keys=["document.input_ids", "document.attention_mask"],
        )

        hq = model(
            batch=batch,
            model_key="question",
        )  # [bs, h]
        he = model(
            batch=doc_batch,
            model_key="document",
        )  # [bs * n_docs, h]

        return {
            "hq": hq,
            "he": he,
        }

    def forward_end(self, output: Batch, split: Split) -> Any:
        """
        gather h_q and h_e from all devices to the device 0
        and compute the similarity matrix between the questions and all the documents.
        This results in a matrix of shape [batch_size, batch_size * n_docs].
        """

        # compute the scoring matrix
        hq, he = (output.pop(k) for k in ["hq", "he"])
        score_matrix = self.similarity(hq, he)  # [bs x bs*n_docs]

        # compute targets
        # assuming the target document is the first of each group
        # the targets are:
        # * 0*n_dods for the first `n_docs` items
        # * 1*n_dods for the following `n_docs` items
        # * ...
        n_docs = he.shape[0] // hq.shape[0]
        targets = n_docs * (
            torch.arange(start=0, end=len(score_matrix))
            .long()
            .to(score_matrix.device)
        )

        # compute the loss an prepare the output
        loss = F.cross_entropy(score_matrix, targets, reduction="mean")
        output["loss"] = loss.mean()
        output["n_options"] = score_matrix.shape[1]  # store the batch size
        output["logits"] = score_matrix
        output["targets"] = targets

        # update the metrics and return
        self.update_metrics(output, split)
        output.pop("logits")
        output.pop("targets")
        return output
