from typing import Any
from typing import Dict
from typing import Optional

from datasets import Split
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torchmetrics import Metric
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy
from torchmetrics.classification import F1
from torchmetrics.classification import Precision
from torchmetrics.classification import Recall

from .abstract import Evaluator


class MultipleChoiceQaMaximumLikelihood(Evaluator):
    """
    Evaluates the Reader model `p(a_i | q, e, A)` using maximum likelihood estimation
    in a multiple choice QA context (A = [a_1,...a_P]). The loss is defined as:

        L =  sum_p log p(a_p | q, e, A) 1(p = a)

    where a is the index of the true answer.
    """

    # TODO: formalize the Metrics logic (when to compute and log)
    text_key = "question"
    _required_eval_feature_names = [
        "answer_idx",
    ]

    def __init__(self, n_choices: int):
        super().__init__()
        metric_kwargs = {"compute_on_step": False}

        def gen_metric(split):
            return MetricCollection(
                [
                    Accuracy(**metric_kwargs),
                    # F1(**metric_kwargs),
                    # Recall(**metric_kwargs),
                    # Precision(**metric_kwargs),
                ],
                prefix="",
            )

        self.metrics = nn.ModuleDict(
            {
                f"_{split}": gen_metric(split)
                for split in [Split.TRAIN, Split.VALIDATION, Split.TEST]
            }
        )

    def get_metric(self, split: str) -> Metric:
        return self.metrics[f"_{split}"]

    def forward(
        self, model: nn.Module, batch: Any, split: str, **kwargs: Any
    ) -> Dict[str, Tensor]:
        self.check_batch_type(batch)
        self.check_feature_names(batch)

        logits: Tensor = model(batch)
        targets: Tensor = batch["answer_idx"]
        loss = F.cross_entropy(logits, targets, reduction="mean")
        self.get_metric(split).update(logits.argmax(-1), targets)
        return {"loss": loss}

    def check_feature_names(self, batch):
        for f in self._required_eval_feature_names:
            assert (
                f in batch.keys()
            ), f"The feature {f} is required for evaluation."

    def reset_metrics(self, split: Optional[str] = None) -> None:
        """reset the metrics"""
        if split is None:
            map(lambda m: m.reset(), self.metrics.values())
        else:
            self.get_metric(split).reset()

    def compute_metrics(
        self, split: Optional[str] = None
    ) -> Dict[str, Tensor]:
        """Compute the metrics"""
        if split is not None:
            metrics = [self.get_metric(split)]
        else:
            metrics = self.metrics.values()

        output = {}
        for metric in metrics:
            output.update(**metric.compute())

        return output
