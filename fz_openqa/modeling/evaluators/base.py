import collections
from typing import Any
from typing import Optional

import torch
from datasets import Split
from torch import nn
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy

from fz_openqa.modeling.evaluators.metrics import SplitMetrics
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.functional import batch_reduce


class BaseEvaluator(nn.Module):
    """
    The Evaluator:
        1. computes the loss
        2. computes and track the metrics (accuracy, F1, ...) using `SplitMetrics`

    !! Important !!
    Metrics needs to be updated in the call of `_step_end` in the LightningModule in order to avoid errors with dp.
    Therefore all the update steps need to be implemented in `update_metrics`, which is subsequently called in
    BaseModel._step_end() on device 0.
    See https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metrics-in-dataparallel-dp-mode
    """

    # named of the required features
    _required_eval_feature_names = [
        "input_ids",
        "attention_mask",
        "labels",
    ]

    def __init__(self, prefix: str = ""):
        """Initialize a Metric for each split=train/validation/test"""
        super().__init__()
        self.init_metrics(prefix=prefix)

    def init_metrics(self, prefix: str):
        metric_kwargs = {"compute_on_step": False, "dist_sync_on_step": True}

        def gen_metric():
            return MetricCollection([Accuracy(**metric_kwargs)], prefix=prefix)

        self.metrics = SplitMetrics(gen_metric)

    def forward(
        self, model: nn.Module, batch: Batch, split: str, **kwargs: Any
    ) -> Batch:
        """Compute the forward pass of the model and return output
        Return a dictionary output with at least the key 'loss' and the data
        necessary to compute the metrics, unless the loss is explicitly computed in the
        `post_forward` method.

        This step will be computed in the `*_step()` method of the ligthning module:
         the data is processed separately on each device.

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

    def forward_end(self, output: Batch, split: Split) -> Any:
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
        output.pop("preds", None)
        output.pop("logits", None)
        output.pop("targets", None)
        return output

    def update_metrics(self, output: Batch, split: Split) -> None:
        """update the metrics of the given split."""
        logits, targets = (output[k] for k in ("logits", "targets"))
        self.metrics.update(split, logits, targets)

    def reset_metrics(self, split: Optional[Split] = None) -> None:
        """
        Reset the metrics corresponding to `split` if provided, else
        reset all the metrics.
        """
        self.metrics.reset(split)

    def compute_metrics(self, split: Optional[Split] = None) -> Batch:
        """
        Compute the metrics for the given `split` else compute the metrics for all splits.
        The metrics are return after computation.
        """
        return self.metrics.compute(split)

    def check_feature_names(self, batch: Batch) -> None:
        """
        Check that the batch input has the right keys.
        Potentially raise an error.
        """
        for f in self._required_eval_feature_names:
            assert (
                f in batch.keys()
            ), f"The feature {f} is required for evaluation."

    def check_batch_type(self, batch: Batch) -> None:
        """
        Check that the batch input is of the right type.
        Potentially raise an error.
        """
        assert isinstance(
            batch,
            (
                dict,
                collections.OrderedDict,
                collections.UserDict,
            ),
        )
