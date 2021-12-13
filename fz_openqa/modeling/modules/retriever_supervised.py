from typing import Any
from typing import Optional

import torch
from torch.nn import functional as F

from .option_retriever import Similarity
from .utils import check_only_first_doc_positive
from .utils import flatten_first_dims
from fz_openqa.modeling.modules.base import Module
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
        compute_loss_on_device_zero: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        head = next(iter(self.heads.values()))
        self.similarity = Similarity(head.id)
        self.compute_loss_on_device_zero = compute_loss_on_device_zero

    def _init_metrics(self, prefix: str):
        """Initialize the metrics for each split."""
        self.metrics = self._get_base_metrics(prefix=prefix, topk=[None, 5, 10, 20, 50, 100])

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
        output = self._forward(batch, _compute_similarity=False, **kwargs)
        if not self.compute_loss_on_device_zero:
            output = self._compute_loss(output)
        return output

    def _reduce_step_output(self, output: Batch) -> Batch:
        """
        if `self.self.compute_loss_on_device_zero` is False, gather
        `h_q` and `h_d` from all devices to the device 0 and compute the similarity matrix
        between the questions and all the documents.
        Else, gather logits and targets + average losses.
        """

        if self.compute_loss_on_device_zero:
            output = self._compute_loss(output)

        # average losses
        for k in ["loss"]:
            y = output.get(k, None)
            if y is not None:
                output[k] = y.mean()

        return output

    def _compute_loss(self, output):
        """This results in a matrix of shape [batch_size, batch_size * n_docs]."""
        hq, hd = (output.pop(k) for k in ["_hq_", "_hd_"])

        # define the score matrix of shape [batch_size, batch_size * n_docs]
        score_matrix = self.compute_similarity(hq, hd)

        # generate targets [0, 1*n_docs, ..., batch_size * n_docs]
        targets = self._generate_targets(
            len(score_matrix),
            n_docs=hd.shape[0] // hq.shape[0],
            device=hd.device,
        )

        # compute the loss an prepare the output
        loss = F.cross_entropy(score_matrix, targets, reduction="none")
        output["loss"] = loss.mean()
        output["n_options"] = score_matrix.shape[1]
        output["_logits_"] = score_matrix.detach()
        output["_targets_"] = targets.detach()

        return output

    def compute_similarity(self, hq: torch.Tensor, hd: torch.Tensor) -> torch.Tensor:
        """
        Compute the similarity between the question and all the documents.
        """
        if self.similarity == Similarity.CLS:
            return torch.einsum("nh, mh -> nm", hq, hd)
        elif self.similarity == Similarity.COLBERT:
            scores = torch.einsum("nqh, mdh -> nmqd", hq, hd)
            max_scores, _ = torch.max(scores, dim=-1)
            return max_scores.mean(-1)
        else:
            raise ValueError(f"Unknown similarity: {self.similarity}")

    def _generate_targets(self, batch_size, *, n_docs: int, device: torch.device):
        """Generate targets. Assuming the target document is the first
        of each group, the targets are [0, 1*n_docs, ..., batch_size * n_docs]."""
        return torch.arange(start=0, end=n_docs * batch_size, step=n_docs, device=device).long()
