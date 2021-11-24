from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F

from .utils import flatten_first_dims
from fz_openqa.modeling.modules.base import Module
from fz_openqa.utils.datastruct import Batch


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
    ]

    # prefix for the logged metrics
    task_id: Optional[str] = "reader"

    # metrics to display in the progress bar
    pbar_metrics = [
        "train/reader/logp",
        "validation/reader/logp",
        "train/reader/Accuracy",
        "validation/reader/Accuracy",
    ]

    # require heads
    _required_heads = ["question", "document"]

    def _init_metrics(self, prefix: str):
        """Initialize the metrics for each split."""
        self.metrics = self._get_base_metrics(prefix=f"{prefix}")

    def _forward_as_flattened(self, batch: Batch, **kwargs: Any) -> Batch:

        d_shape = batch["document.input_ids"].shape
        q_shape = batch["question.input_ids"].shape
        if not d_shape[:2] == q_shape[:2]:
            raise ValueError(
                f"Expected 2 first dimensions to be equal, "
                f"got documents of shape: {d_shape} and "
                f"questions of shape{q_shape}"
            )

        # flatten the batch_size and n_options and n_docs dimensions
        d_batch = flatten_first_dims(
            batch,
            n_dims=3,
            keys=["document.input_ids", "document.attention_mask"],
        )

        # flatten the batch_size and n_options dimensions
        q_batch = flatten_first_dims(
            batch,
            n_dims=2,
            keys=["question.input_ids", "question.attention_mask"],
        )

        # process the document with the backbones
        hd = self._backbone(d_batch, prefix="document", head="document", **kwargs)
        hq = self._backbone(q_batch, prefix="question", head="question", **kwargs)

        # reshape and return
        hd = hd.reshape(*d_shape[:3], *hd.shape[1:])
        hq = hq.reshape(*q_shape[:2], *hq.shape[1:])
        return {"_hd_": hd, "_hq_": hq}

    def _forward(self, batch: Batch, **kwargs) -> Batch:

        # process the batch using BERT and the heads
        output = self._forward_as_flattened(batch, **kwargs)
        hq, hd = (output[k] for k in ["_hq_", "_hd_"])

        # compute the reader logits
        s_reader = torch.einsum("bah, badh -> bad", hq, hd)

        # compute the retriever logits
        s_retriever = s_reader

        # answer likelihood p(a | q, D, A)
        logp_a__d = s_reader.log_softmax(dim=1)
        targets = batch["answer.target"]
        targets_ = targets.unsqueeze(1).expand(-1, s_reader.shape[2])
        logp_a_star__d = F.cross_entropy(logp_a__d, targets_, reduction="none")

        # retriever likelihood
        logp_d = s_retriever.log_softmax(dim=2)
        logp_d_a_star = torch.gather(logp_d, dim=1, index=targets[:, None, None])

        # likelihood
        logp_a = (logp_a__d * logp_d.exp()).sum(-1)
        logp_a_star = torch.gather(logp_a, dim=1, index=targets[:, None])

        # gradients / loss
        loss_a = logp_a_star__d
        loss_b = (logp_a_star__d.detach() * logp_d_a_star).sum(1)
        loss = -1 * (loss_a + loss_b).mean(-1)

        return {
            "loss": loss,
            "logp": logp_a_star.detach(),
            "_logits_": logp_a.detach(),
            "_targets_": targets.detach(),
        }

    def _step(self, batch: Batch, **kwargs: Any) -> Batch:
        """
        Compute the forward pass for the question and the documents.
        """
        # check features, check that the first document of each question is positive
        return self._forward(batch, **kwargs)

    def _reduce_step_output(self, output: Batch) -> Batch:
        """
        Gather losses and logits from all devides and return
        """

        # average losses
        for k in ["loss", "logp"]:
            y = output.get(k, None)
            if y is not None:
                output[k] = y.mean()

        return output
