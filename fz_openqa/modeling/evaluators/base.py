import collections
import re
from typing import Any
from typing import List
from typing import Optional

import torch
from datasets import Split
from torch import nn
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy

from fz_openqa.modeling.backbone import Backbone
from fz_openqa.modeling.evaluators.metrics import SplitMetrics
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.functional import batch_reduce

FEATURE_PATTERN = re.compile("^_{1}[a-zA-Z0-9]+_{1}$")


class Evaluator(nn.Module):
    """
    The Evaluator:
        1. computes the loss
        2. computes and track the metrics (accuracy, F1, ...) using `SplitMetrics`

    !! Important !!
    Metrics needs to be updated in the call of `_step_end` in the
    LightningModule in order to avoid errors with dp.
    Therefore all the update steps need to be implemented in `update_metrics`,
    which is subsequently called in
    BaseModel.step_end() on device 0.
    See https://torchmetrics.readthedocs.io/en/stable/pages/
        overview.html#metrics-in-dataparallel-dp-mode
    """

    # name of the features required for a forward pass
    _required_feature_names = [
        "input_ids",
        "attention_mask",
    ]

    # named of the features required for evaluation
    _required_eval_feature_names = [
        "input_ids",
        "attention_mask",
        "labels",
    ]

    def __init__(self, *, backbone: Backbone, prefix: str = ""):
        """Initialize a Metric for each split=train/validation/test"""
        super().__init__()
        self.backbone = backbone
        self.init_metrics(prefix=prefix)

    def init_metrics(self, prefix: str):
        metric_kwargs = {"compute_on_step": False, "dist_sync_on_step": True}

        def init_metric():
            return MetricCollection([Accuracy(**metric_kwargs)], prefix=prefix)

        self.metrics = SplitMetrics(init_metric)

    def _format_exception(
        self, batch: Batch, required_feature_names: List[str]
    ):
        missing = [f for f in required_feature_names if f not in batch]
        return (
            f"{type(self).__name__} requires features {required_feature_names}, "
            f"features={missing} are missing in batch with keys={list(batch.keys())}."
        )

    def _check_features(self, batch, required_feature_names: List[str]):
        assert all(f in batch for f in required_feature_names), self._fmt_ex(
            batch, required_feature_names
        )

    def forward(self, batch, **kwargs):
        """Compute the forward pass of the model, does not require targets,
        it can be used for inference."""
        self._check_features(batch, self._required_feature_names)
        return self.backbone(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            **kwargs,
        )

    def evaluate(self, batch: Batch, **kwargs):
        """
        Evaluate the model (step + step end) given a batch of data
        with targets
        """
        output = self.step(batch, None, **kwargs)
        return self.step_end(output, None, update_metrics=False)

    def step(
        self, batch: Batch, split: Optional[Split] = None, **kwargs: Any
    ) -> Batch:
        """Compute the forward pass of the model and return output
        Return a dictionary output with at least the key 'loss' and the data
        necessary to compute the metrics, unless the loss is explicitly
        computed in the `post_forward` method.

        This step will be computed in the `*_step()` method of the
        ligthning module: the data is processed separately on each device.

        The torchmetric `Metric.update()` method should not be called here.
        See `post_forward` instead.

        Implement `_step` for each sub-class.
        """
        self._check_features(batch, self._required_eval_feature_names)
        return self._step(batch, split, **kwargs)

    def _step(
        self, batch: Batch, split: Optional[Split] = None, **kwargs: Any
    ) -> Batch:
        raise NotImplementedError

        # example
        logits = self.forward(batch, **kwargs)
        nll = torch.nn.functional.cross_entropy(
            logits, batch["labels"], reduce="none"
        )

        # register internal values (which should not be passed to the pl module) using _<name>_.
        return {
            "loss": batch_reduce(nll, op=torch.mean),
            "_logits_": logits,
            "_targets_": batch["labels"],
        }

    def step_end(
        self,
        output: Batch,
        split: Optional[Split],
        update_metrics: bool = True,
    ) -> Any:
        """Apply a post-processing step to the forward method.
        The output is the output of the forward method.

        This method is called after the `output` has been gathered
        from each device. This method must aggregate the loss across
        devices.

        torchmetrics update() calls should be placed here.
        The output must at least contains the `loss` key.

        Implement `_reduce_step_output` for each sub-class.
        """

        # reduce tensors gathered from parallel `step()` calls
        output = self._reduce_step_output(output)

        # update the metrics
        if update_metrics:
            assert split is not None
            self.update_metrics(output, split)

        # filter internal
        return self._filter_features_from_output(output)

    @staticmethod
    def _filter_features_from_output(output: Batch) -> Batch:
        """filter the internal values such as _logits_ or _targets_"""
        return {
            k: v for k, v in output.items() if not re.match(FEATURE_PATTERN, k)
        }

    def _reduce_step_output(self, output: Batch) -> Batch:
        raise NotImplementedError
        # example
        output["loss"] = output["loss"].mean()

    def update_metrics(self, output: Batch, split: Split) -> None:
        """update the metrics of the given split."""
        logits, targets = (output[k] for k in ("_logits_", "_targets_"))
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
