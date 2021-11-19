# Metrics

> Auto-generated documentation for [fz_openqa.modeling.modules.metrics](blob/master/fz_openqa/modeling/modules/metrics.py) module.

- [Fz-openqa](../../../README.md#fz-openqa-index) / [Modules](../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../index.md#fz-openqa) / [Modeling](../index.md#modeling) / [Modules](index.md#modules) / Metrics
    - [NestedMetricCollections](#nestedmetriccollections)
        - [NestedMetricCollections().compute](#nestedmetriccollectionscompute)
        - [NestedMetricCollections().reset](#nestedmetriccollectionsreset)
        - [NestedMetricCollections().update](#nestedmetriccollectionsupdate)
    - [SafeMetricCollection](#safemetriccollection)
        - [SafeMetricCollection().compute](#safemetriccollectioncompute)
        - [SafeMetricCollection().update](#safemetriccollectionupdate)
    - [SplitMetrics](#splitmetrics)
        - [SplitMetrics().compute](#splitmetricscompute)
        - [SplitMetrics().reset](#splitmetricsreset)
        - [SplitMetrics.safe_compute](#splitmetricssafe_compute)
        - [SplitMetrics().update](#splitmetricsupdate)
    - [is_computable](#is_computable)

## NestedMetricCollections

[[find in source code]](blob/master/fz_openqa/modeling/modules/metrics.py#L97)

```python
class NestedMetricCollections(MetricCollection):
    def __init__(metrics: Dict[str, MetricCollection]):
```

A class that allows handling multiple sub-MetricCollections, each of them index by a key.
Only the signature of the update method changes, which requires a dictionary of tuples as input.

### NestedMetricCollections().compute

[[find in source code]](blob/master/fz_openqa/modeling/modules/metrics.py#L111)

```python
def compute() -> Any:
```

### NestedMetricCollections().reset

[[find in source code]](blob/master/fz_openqa/modeling/modules/metrics.py#L119)

```python
def reset() -> None:
```

### NestedMetricCollections().update

[[find in source code]](blob/master/fz_openqa/modeling/modules/metrics.py#L107)

```python
def update(values=Dict[str, Tuple[Tensor]]) -> None:
```

## SafeMetricCollection

[[find in source code]](blob/master/fz_openqa/modeling/modules/metrics.py#L74)

```python
class SafeMetricCollection(MetricCollection):
```

A safe implementation of MetricCollection, so top-k accuracy  won't
raise an error if the batch size is too small.

### SafeMetricCollection().compute

[[find in source code]](blob/master/fz_openqa/modeling/modules/metrics.py#L89)

```python
def compute() -> Dict[str, Any]:
```

### SafeMetricCollection().update

[[find in source code]](blob/master/fz_openqa/modeling/modules/metrics.py#L80)

```python
def update(*args: Any, **kwargs: Any) -> None:
```

## SplitMetrics

[[find in source code]](blob/master/fz_openqa/modeling/modules/metrics.py#L22)

```python
class SplitMetrics(nn.Module):
    def __init__(init_metric: [None, Metric]):
```

Define a metric for each split

### SplitMetrics().compute

[[find in source code]](blob/master/fz_openqa/modeling/modules/metrics.py#L57)

```python
def compute(split: Optional[Split] = None) -> Batch:
```

Compute the metrics for the given `split` else compute the metrics for all splits.
The metrics are return after computation.

#### See also

- [Batch](../../utils/datastruct.md#batch)

### SplitMetrics().reset

[[find in source code]](blob/master/fz_openqa/modeling/modules/metrics.py#L41)

```python
def reset(split: Optional[Split]):
```

### SplitMetrics.safe_compute

[[find in source code]](blob/master/fz_openqa/modeling/modules/metrics.py#L51)

```python
@staticmethod
def safe_compute(metric: MetricCollection) -> Batch:
```

equivalent to `MetricCollection.compute`,
but filtering metrics where metric.mode is not set (which happens if there was no update)

#### See also

- [Batch](../../utils/datastruct.md#batch)

### SplitMetrics().update

[[find in source code]](blob/master/fz_openqa/modeling/modules/metrics.py#L47)

```python
def update(split: Split, *args: Tuple[torch.Tensor]) -> None:
```

update the metrics of the given split.

## is_computable

[[find in source code]](blob/master/fz_openqa/modeling/modules/metrics.py#L17)

```python
def is_computable(m: Metric):
```

check if one can call .compute() on metric
