from typing import Any

import torch
from datasets import Split
from torch import nn
from torch.nn import functional as F
from torchmetrics.classification import Accuracy

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

        def gen_metric():
            """generate a collection of topk accuracies."""
            return SafeMetricCollection(
                {
                    _name(k): Accuracy(top_k=k, **metric_kwargs)
                    for k in [None, 5, 10, 20, 50, 100]
                },
                prefix=prefix,
            )

        self.metrics = SplitMetrics(gen_metric)

    def forward(
        self, model: nn.Module, batch: Batch, split: Split, **kwargs: Any
    ) -> Batch:
        self.check_batch_type(batch)
        self.check_feature_names(batch)

        # check that the first document of each question is positive
        assert torch.all(batch["document.is_positive"][:, 0] == 1)
        if batch["document.is_positive"].shape[1] > 1:
            assert torch.all(batch["document.is_positive"][:, 1:] == 0)

        # flatten documents inputs
        doc_batch = {
            k: batch[k].view(-1, *batch[k].shape[2:])
            for k in ["document.input_ids", "document.attention_mask"]
        }

        hq = model(
            batch=batch,
            model_key="question",
        )  # [bs, h]
        he = model(
            batch=doc_batch,
            model_key="document",
        )  # [bs, h]

        return {
            "hq": hq,
            "he": he,
        }

    def forward_end(self, output: Batch, split: str) -> Any:
        """Apply a post-processing step to the forward method.
        The output is the output of the forward method.

        This method is called after the `output` has been gathered
        from each device. This method must aggregate the loss across
        devices.

        torchmetrics update() calls should be placed here.
        The output must at least contains the `loss` key.
        """
        hq, he = (output.pop(k) for k in ["hq", "he"])

        # compute logits
        logits = self.similarity(hq, he)  # [bs x bs*n_docs]

        # compute targets
        # assuming the target document is the first of each group
        n_docs = he.shape[0] // hq.shape[0]
        targets = n_docs * (
            torch.arange(start=0, end=len(logits)).long().to(logits.device)
        )
        loss = F.cross_entropy(logits, targets, reduction="mean")
        output["loss"] = loss.mean()
        output["n_options"] = logits.shape[1]  # store the batch size
        output["logits"] = logits
        output["targets"] = targets
        self.update_metrics(output, split)
        output.pop("logits")
        output.pop("targets")
        return output
