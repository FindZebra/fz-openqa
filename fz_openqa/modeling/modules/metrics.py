from copy import deepcopy
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from datasets import Split
from torch import nn
from torch import Tensor
from torchmetrics import Accuracy
from torchmetrics import Metric
from torchmetrics import MetricCollection

from fz_openqa.utils.datastruct import Batch


def is_computable(m: Metric):
    """check if one can call .compute() on metric"""
    return not isinstance(m, Accuracy) or m.mode is not None


class SplitMetrics(nn.Module):
    """Define a metric for each split"""

    def __init__(self, metric: Union[MetricCollection, Metric]):
        super(SplitMetrics, self).__init__()

        self.train_metric = deepcopy(metric)
        self.valid_metric = deepcopy(metric)
        self.test_metric = deepcopy(metric)

        self.metrics = {
            f"_{Split.TRAIN}": self.train_metric,
            f"_{Split.VALIDATION}": self.valid_metric,
            f"_{Split.TEST}": self.test_metric,
        }

    def __getitem__(self, split: Split) -> Metric:
        return self.metrics[f"_{split}"]

    def reset(self, split: Optional[Split]):
        if split is None:
            map(lambda m: m.reset(), self.metrics.values())
        else:
            self[split].reset()

    def update(self, split: Split, *args: Tuple[torch.Tensor]) -> None:
        """update the metrics of the given split."""
        self[split].update(*args)

    @staticmethod
    def safe_compute(metric: MetricCollection) -> Batch:
        """equivalent to `MetricCollection.compute`,
        but filtering metrics where metric.mode is not set (which happens if there was no update)"""
        return {k: m.compute() for k, m in metric.items() if is_computable(m)}

    def compute(self, split: Optional[Split] = None) -> Batch:
        """
        Compute the metrics for the given `split` else compute the metrics for all splits.
        The metrics are return after computation.
        """
        if split is not None:
            metrics = [self[split]]
        else:
            metrics = self.metrics.values()

        output = {}
        for metric in metrics:
            output.update(**self.safe_compute(metric))

        return output


class SafeMetricCollection(MetricCollection):
    """
    A safe implementation of MetricCollection, so top-k accuracy  won't
    raise an error if the batch size is too small.
    """

    def update(self, *args: Any, **kwargs: Any) -> None:
        for _, m in self.items(keep_base=True):
            preds, targets = args
            if isinstance(m, Accuracy) and m.top_k is not None and preds.shape[-1] <= m.top_k:
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


class NestedMetricCollections(MetricCollection):
    """
    A class that allows handling multiple sub-MetricCollections, each of them index by a key.
    Only the signature of the update method changes, which requires a dictionary of tuples as input.
    """

    def __init__(self, metrics: Dict[str, MetricCollection]):
        nn.Module.__init__(self)
        self.metrics = nn.ModuleDict(metrics)

    def update(self, values=Dict[str, Tuple[Tensor]]) -> None:
        for k, v in values.items():
            self.metrics[k].update(*v)

    def compute(self) -> Any:
        return {
            k: v
            for metric in self.metrics.values()
            if next(iter(metric.values())).mode is not None
            for k, v in metric.compute().items()
        }

    def reset(self) -> None:
        for metric in self.metrics.values():
            metric.reset()
