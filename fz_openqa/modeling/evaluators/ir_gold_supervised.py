from typing import Any
from typing import Dict

import torch
from datasets import Split
from torch import nn
from torch.nn import functional as F
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy

from fz_openqa.modeling.evaluators.base import Evaluator
from fz_openqa.modeling.similarities import Similarity
from fz_openqa.utils.datastruct import Batch


class SafeMetricCollection(MetricCollection):
    """A safe implementation of MetricCollection, so top-k accuarcy  won't
    raise an error if the batch size is too small."""

    def update(self, *args: Any, **kwargs: Any) -> None:
        for _, m in self.items(keep_base=True):
            preds, targets = args
            if (
                isinstance(m, Accuracy)
                and m.top_k is not None
                and preds.shape[-1] <= m.top_k
            ):
                pass
            else:
                m_kwargs = m._filter_kwargs(**kwargs)
                m.update(preds, targets, **m_kwargs)

    def compute(self) -> Dict[str, Any]:
        return {
            k: m.compute()
            for k, m in self.items()
            if not isinstance(m, Accuracy) or m.mode is not None
        }


class InformationRetrievalGoldSupervised(Evaluator):
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

        # todo: check if `dist_sync_on_step` is necessary
        # see https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metrics-in-dataparallel-dp-mode

        metric_kwargs = {"compute_on_step": False, "dist_sync_on_step": True}

        def _name(k):
            return f"top{k}_Accuracy" if k is not None else "Accuracy"

        def gen_metric_one_split():
            return SafeMetricCollection(
                {
                    _name(k): Accuracy(top_k=k, **metric_kwargs)
                    for k in [None, 5, 10, 20, 50, 100]
                },
                prefix=prefix,
            )

        self.metrics = nn.ModuleDict(
            {
                f"_{split}": gen_metric_one_split()
                for split in [Split.TRAIN, Split.VALIDATION, Split.TEST]
            }
        )

    def forward(
        self, model: nn.Module, batch: Batch, split: str, **kwargs: Any
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
