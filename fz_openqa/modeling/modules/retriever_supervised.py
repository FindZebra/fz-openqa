from typing import Any
from typing import Optional
from typing import Union

import torch
from omegaconf import DictConfig
from torch.nn import functional as F
from torchmetrics.classification import Accuracy

from ...utils import maybe_instantiate
from .utils import check_only_first_doc_positive
from .utils import flatten_first_dims
from fz_openqa.modeling.modules.base import Module
from fz_openqa.modeling.modules.metrics import SafeMetricCollection
from fz_openqa.modeling.modules.metrics import SplitMetrics
from fz_openqa.modeling.similarities import DotProduct
from fz_openqa.modeling.similarities import Similarity
from fz_openqa.utils.datastruct import Batch


class RetrieverSupervised(Module):
    # features are not all required in forward() depending on the context,
    # so we pass this check for that model
    _required_feature_names = []

    _required_eval_feature_names = [
        "question.input_ids",
        "question.attention_mask",
        "document.input_ids",
        "document.attention_mask",
        "document.match_score",
    ]

    # prefix for the logged metrics
    task_id: Optional[str] = "retriever"

    # metrics to display in the progress bar
    pbar_metrics = [
        "train/retriever/Accuracy",
        "validation/retriever/Accuracy",
        "validation/retriever/top10_Accuracy",
        "validation/retriever/n_options",
    ]

    # require heads
    _required_heads = ["question", "document"]

    def __init__(
        self,
        similarity: Union[DictConfig, Similarity] = DotProduct(),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.similarity = maybe_instantiate(similarity)

    def _init_metrics(self, prefix: str):
        """Initialize the metrics for each split."""
        metric_kwargs = {"compute_on_step": False, "dist_sync_on_step": True}

        def _name(k):
            return f"top{k}_Accuracy" if k is not None else "Accuracy"

        def init_metric():
            """generate a collection of topk accuracies."""
            return SafeMetricCollection(
                {_name(k): Accuracy(top_k=k, **metric_kwargs) for k in [None, 5, 10, 20, 50, 100]},
                prefix=prefix,
            )

        self.metrics = SplitMetrics(init_metric)

    def _forward_question(self, batch: Batch, **kwargs: Any) -> Batch:
        assert batch["question.input_ids"].dim() == 2
        hq = self._backbone(batch, prefix="question", head="question", **kwargs)
        return {"_hq_": hq}

    def _forward_document(self, batch: Batch, **kwargs: Any) -> Batch:
        # check input dimensions
        input_dim = batch["document.input_ids"].dim()
        msg = (
            f"Expected input_ids to be of shape [batch_size, n_docs, seq_len], "
            f"or [batch_size, seq_len], got {input_dim} dimensions"
        )
        assert input_dim in {2, 3}, msg

        # potentially flatten the batch_size and n_docs dimensions
        if input_dim == 3:
            batch = flatten_first_dims(
                batch,
                n_dims=2,
                keys=["document.input_ids", "document.attention_mask"],
            )

        # process the document with the backbone and returns
        hd = self._backbone(batch, prefix="document", head="document", **kwargs)
        return {"_hd_": hd}

    def _forward(self, batch: Batch, _compute_similarity: bool = True, **kwargs) -> Batch:

        # check inputs and set arguments accordingly
        assert "question.input_ids" in batch or "document.input_ids" in batch
        if not ("question.input_ids" in batch and "document.input_ids" in batch):
            _compute_similarity = False

        output = {}
        if "question.input_ids" in batch:
            output.update(self._forward_question(batch, **kwargs))

        if "document.input_ids" in batch:
            output.update(self._forward_document(batch, **kwargs))

        if _compute_similarity:
            output["score"] = self.similarity(output["_hq_"], output["_hd_"])

        return output

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
        return self._forward(batch, _compute_similarity=False, **kwargs)

    def _reduce_step_output(self, output: Batch) -> Batch:
        """
        gather h_q and h_e from all devices to the device 0
        and compute the similarity matrix between the questions and all the documents.
        This results in a matrix of shape [batch_size, batch_size * n_docs].
        """
        # compute the scoring matrix
        hq, hd = (output.pop(k) for k in ["_hq_", "_hd_"])
        score_matrix = self.similarity(hq, hd)  # [bs x bs*n_docs]
        targets = self._generate_targets(
            len(score_matrix),
            n_docs=hd.shape[0] // hq.shape[0],
            device=hd.device,
        )

        # compute the loss an prepare the output
        loss = F.cross_entropy(score_matrix, targets, reduction="mean")
        output["loss"] = loss.mean()
        output["n_options"] = score_matrix.shape[1]
        output["_logits_"] = score_matrix
        output["_targets_"] = targets

        return output

    def _generate_targets(self, batch_size, *, n_docs: int, device: torch.device):
        """Generate targets. Assuming the target document is the first
        of each group, the targets are:
          * 0*n_docs for the first `n_docs` items
          * 1*n_docs for the following `n_docs` items
          * ..."""
        one_to_bs = torch.arange(start=0, end=batch_size, device=device).long()
        return n_docs * one_to_bs
