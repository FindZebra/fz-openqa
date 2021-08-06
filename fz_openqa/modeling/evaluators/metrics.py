from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import torch
from datasets import Split
from torch import nn
from torchmetrics import Accuracy
from torchmetrics import Metric
from torchmetrics import MetricCollection

from fz_openqa.utils.datastruct import Batch


class SplitMetrics(nn.Module):
    """Define a metric for each split"""

    def __init__(self, init_metric: [None, Metric]):
        super(SplitMetrics, self).__init__()

        self.train_metric = init_metric()
        self.valid_metric = init_metric()
        self.test_metric = init_metric()

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
            self.get_metric(split).reset()

    def update(self, split: Split, *args: Tuple[torch.Tensor]) -> None:
        """update the metrics of the given split."""
        self[split].update(*args)

    def compute(self, split: Optional[Split] = None) -> Batch:
        """
        Compute the metrics for the given `split` else compute the metrics for all splits.
        The metrics are return after computation.
        """
        if split is not None:
            metrics = [self.get_metric(split)]
        else:
            metrics = self.metrics.values()

        output = {}
        for metric in metrics:
            output.update(**metric.compute())

        return output


class SafeMetricCollection(MetricCollection):
    """
    A safe implementation of MetricCollection, so top-k accuracy  won't
    raise an error if the batch size is too small.
    """

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
