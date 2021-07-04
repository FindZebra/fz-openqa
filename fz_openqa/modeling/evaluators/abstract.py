import collections
from typing import Any
from typing import Optional

import torch
from datasets import Split
from torch import nn
from torchmetrics import Metric
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy

from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.functional import batch_reduce


class Evaluator(nn.Module):
    """A base class to evaluate model's output given a model and a batch of data.
    Track and compute metrics. Metrics needs to be updated in the call of `_step_end` in the
    LightningModule in order to avoid errors with dp. Therefore all the update steps will be
    implemented in `update_metrics`.
    See https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metrics-in-dataparallel-dp-mode
    """

    _required_eval_feature_names = [
        "input_ids",
        "attention_mask",
        "labels",
    ]

    def __init__(self, prefix: str = ""):
        super().__init__()
        self.init_metrics(prefix)

    def get_metric(self, split: str) -> Metric:
        return self.metrics[f"_{split}"]

    def init_metrics(self, prefix: str):
        """Initialize the metrics for each split."""

        # todo: check if `dist_sync_on_step` is necessary
        # see https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metrics-in-dataparallel-dp-mode

        metric_kwargs = {"compute_on_step": False, "dist_sync_on_step": True}

        def gen_metric_one_split():
            return MetricCollection([Accuracy(**metric_kwargs)], prefix=prefix)

        self.metrics = nn.ModuleDict(
            {
                f"_{split}": gen_metric_one_split()
                for split in [Split.TRAIN, Split.VALIDATION, Split.TEST]
            }
        )

    def forward(
        self, model: nn.Module, batch: Batch, split: str, **kwargs: Any
    ) -> Batch:
        """Compute the forward pass of the model and return output
        Return a dictionary output with at least the key 'loss' and the data
        necessary to compute the metrics, unless the loss is explicitly computed in the
        `post_forward` method.

        This step will be computed in the `*_step()` method of the ligthning module.
        Hence the data is processed separately on each device.

        The torchmetric `Metric.update()` method should not be called here. See `post_forward` instead.
        """
        raise NotImplementedError

        # example
        logits = model(batch["input_ids"])
        nll = torch.nn.functional.cross_entropy(
            logits, batch["labels"], reduce="none"
        )

        return {
            "loss": batch_reduce(nll, op=torch.mean),
            "logits": logits,
            "targets": batch["labels"],
        }

    def post_forward(self, output: Batch, split: str) -> Any:
        """Apply a post-processing step to the forward method.
        The output is the output of the forward method.

        This method is called after the `output` has been gathered
        from each device. This method must aggregate the loss across
        devices.

        torchmetrics update() calls should be placed here.
        The output must at least contains the `loss` key.
        """

        output["loss"] = output["loss"].mean()
        self.update_metrics(output, split)
        output.pop("preds")
        output.pop("logits")
        return output

    def update_metrics(self, output: Batch, split: str) -> None:
        """update the metrics of the given split."""
        logits, targets = (output[k] for k in ("logits", "targets"))
        self.get_metric(split).update(logits, targets)

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

    def compute_metrics(self, split: Optional[str] = None) -> Batch:
        """Compute the metrics"""
        if split is not None:
            metrics = [self.get_metric(split)]
        else:
            metrics = self.metrics.values()

        output = {}
        for metric in metrics:
            output.update(**metric.compute())

        return output

    def check_batch_type(self, batch):
        assert isinstance(
            batch,
            (
                dict,
                collections.OrderedDict,
                collections.UserDict,
            ),
        )
