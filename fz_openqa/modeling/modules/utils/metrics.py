from __future__ import annotations

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
from torchmetrics.retrieval import RetrievalMetric

from fz_openqa.utils.datastruct import Batch


def is_computable(m: Metric | nn.Module):
    """check if one can call .compute() on metric"""
    if isinstance(m, Accuracy):
        return m.mode is not None
    elif isinstance(m, RetrievalMetric):
        return len(m.indexes) > 0
    else:
        return True


class SplitMetrics(nn.Module):
    """Define a metric for each split"""

    def __init__(
        self,
        metric: Union[MetricCollection, Metric],
        allowed_splits: Optional[Tuple[Split, ...]] = None,
    ):
        super(SplitMetrics, self).__init__()

        splits = {Split.TRAIN, Split.VALIDATION, Split.TEST}
        if allowed_splits is not None:
            splits = set.intersection(splits, set(allowed_splits))
        self.splits = splits

        self.metrics = nn.ModuleDict({f"_{split}": deepcopy(metric) for split in self.splits})

    def __getitem__(self, split: Split) -> Metric:
        return self.metrics[f"_{split}"]

    def reset(self, split: Optional[Split]):
        if split is None:
            map(lambda m: m.reset(), self.metrics.values())
        elif split in self.splits:
            self[split].reset()
        else:
            pass

    def update(self, split: Split, *args: torch.Tensor | None) -> None:
        """update the metrics of the given split."""
        if split in self.splits:
            self[split].update(*args)
        else:
            pass

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
        if split is not None and split in self.splits:
            metrics = [self[split]]
        else:
            metrics = self.metrics.values()

        output = {}
        for metric in metrics:
            output.update(**self.safe_compute(metric))

        return output


class SafeMetricCollection(MetricCollection):
    """
    A safe implementation of MetricCollection handling multiple failues:
        1. top-k accuracy  won't raise an error if the batch size is too small.
        2. automatically fills the `index` attribute for `RetrievalMetric`
    """

    def update(self, *args: Any, **kwargs: Any) -> None:
        args = list(args)

        # automatically create the index if needed
        if len(args) < 3 and any(isinstance(m, RetrievalMetric) for m in self.values()):
            m = [m for m in self.values() if isinstance(m, RetrievalMetric)][0]
            preds, targets = args
            start_index = 1 + max(i.max() for i in m.indexes) if len(m.indexes) else 0
            indexes = torch.arange(start_index, len(preds) + start_index, device=preds.device)
            indexes = indexes.view(indexes.size(0), *(1 for _ in preds.shape[1:]))
            indexes = indexes.expand_as(preds)
            args = [preds, targets, indexes]

        for _, m in self.items(keep_base=True):

            # handle Accuracy
            if isinstance(m, Accuracy):
                # skip metric if n_documents <= k
                preds, targets, *_ = args
                if m.top_k is not None and preds.shape[-1] <= m.top_k:
                    continue

                m_kwargs = m._filter_kwargs(**kwargs)
                m.update(*args[:2], **m_kwargs)

            # handle Retrieval Metrics
            elif isinstance(m, RetrievalMetric):
                # skip metric if n_documents <= k
                if hasattr(m, "k"):
                    preds, targets, *_ = args
                    if m.k is not None and preds.shape[-1] <= m.k:
                        continue

                m_kwargs = m._filter_kwargs(**kwargs)
                m.update(*args[:3], **m_kwargs)

            else:
                m_kwargs = m._filter_kwargs(**kwargs)
                m.update(*args, **m_kwargs)

    def compute(self) -> Dict[str, Any]:
        return {k: m.compute() for k, m in self.items() if is_computable(m)}


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
