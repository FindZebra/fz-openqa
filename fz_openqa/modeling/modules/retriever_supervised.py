from typing import Any
from typing import Optional

import torch
from torch.nn import functional as F
from torchmetrics.classification import Accuracy

from .utils import check_only_first_doc_positive
from .utils import flatten_first_dims
from fz_openqa.modeling.modules.base import Module
from fz_openqa.modeling.modules.metrics import SafeMetricCollection
from fz_openqa.modeling.modules.metrics import SplitMetrics
from fz_openqa.modeling.similarities import Similarity
from fz_openqa.utils.datastruct import Batch


class RetrieverSupervised(Module):
    _required_feature_names = [
        "question.input_ids",
        "question.attention_mask",
        "document.input_ids",
        "document.attention_mask",
    ]

    _required_eval_feature_names = [
        "document.is_positive",
    ]

    # prefix for the logged metrics
    task_id: Optional[str] = "retrieval"

    # metrics to display in the progress bar
    pbar_metrics = [
        "train/retrieval/Accuracy",
        "validation/retrieval/Accuracy",
        "validation/retrieval/top10_Accuracy",
    ]

    def __init__(self, similarity: Similarity, **kwargs):
        super().__init__(**kwargs)
        self.similarity = similarity

    def _init_metrics(self, prefix: str):
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

    def _forward(self, batch: Batch, reshape: bool = True, **kwargs) -> Batch:
        batch_size, n_documents, *_ = batch["document.input_ids"].shape
        # flatten documents as [batch_size x n_documents]
        batch = flatten_first_dims(
            batch,
            n_dims=2,
            keys=["document.input_ids", "document.attention_mask"],
        )

        # shape: [bs, h]
        hq = self.backbone(
            batch["question.input_ids"],
            attention_mask=batch["question.attention_mask"],
        )
        # shape: # [bs * n_docs, h]
        hd = self.backbone(
            batch["document.input_ids"],
            attention_mask=batch["document.attention_mask"],
        )

        if reshape:
            hd = hd.view(batch_size, n_documents, *hq.shape[1:])

        return {"_hq_": hq, "_hd_": hd}

    def _step(self, batch: Batch, **kwargs: Any) -> Batch:
        """
        Compute the forward pass for the question and the documents.

        The input data is assumed to be of shape:
        batch = {
        'question.input_ids': [batch_size, L_q],
        'document.input_ids': [batch_size, n_docs, L_q]
        """
        # check features, check that the first document of each question is positive
        check_only_first_doc_positive(batch)
        return self._forward(batch, reshape=False, **kwargs)

    def _reduce_step_output(self, output: Batch) -> Batch:
        """
        gather h_q and h_e from all devices to the device 0
        and compute the similarity matrix between the questions and all the documents.
        This results in a matrix of shape [batch_size, batch_size * n_docs].
        """
        # compute the scoring matrix
        hq, hd = (output.pop(k) for k in ["_hq_", "_hd_"])
        score_matrix = self.similarity(hq, hd)  # [bs x bs*n_docs]

        # compute targets
        # assuming the target document is the first of each group
        # the targets are:
        # * 0*n_docs for the first `n_docs` items
        # * 1*n_docs for the following `n_docs` items
        # * ...
        n_docs = hd.shape[0] // hq.shape[0]
        targets = n_docs * (
            torch.arange(start=0, end=len(score_matrix))
            .long()
            .to(score_matrix.device)
        )

        # compute the loss an prepare the output
        loss = F.cross_entropy(score_matrix, targets, reduction="mean")
        output["loss"] = loss.mean()
        output["n_options"] = score_matrix.shape[1]  # store the batch size
        output["_logits_"] = score_matrix
        output["_targets_"] = targets

        return output
