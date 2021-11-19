# Base

> Auto-generated documentation for [fz_openqa.modeling.modules.base](blob/master/fz_openqa/modeling/modules/base.py) module.

- [Fz-openqa](../../../README.md#fz-openqa-index) / [Modules](../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../index.md#fz-openqa) / [Modeling](../index.md#modeling) / [Modules](index.md#modules) / Base
    - [Module](#module)
        - [Module().compute_metrics](#modulecompute_metrics)
        - [Module().evaluate](#moduleevaluate)
        - [Module().forward](#moduleforward)
        - [Module().reset_metrics](#modulereset_metrics)
        - [Module().step](#modulestep)
        - [Module().step_end](#modulestep_end)
        - [Module().update_metrics](#moduleupdate_metrics)
    - [init_default_heads](#init_default_heads)

## Module

[[find in source code]](blob/master/fz_openqa/modeling/modules/base.py#L44)

```python
class Module(nn.Module, ABC):
    def __init__(
        bert: Union[DictConfig, BertPreTrainedModel],
        tokenizer: Union[DictConfig, PreTrainedTokenizerFast],
        heads: Union[DictConfig, Dict[str, Head]],
        prefix: str = '',
    ):
```

A model:
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

#### Attributes

- `pbar_metrics` - metrics to display: `['train/loss', 'train/Accuracy', 'validation/Accuracy']`
- `max_length` - maximum input size: `512`

### Module().compute_metrics

[[find in source code]](blob/master/fz_openqa/modeling/modules/base.py#L274)

```python
def compute_metrics(split: Optional[Split] = None) -> Batch:
```

Compute the metrics for the given `split` else compute the metrics for all splits.
The metrics are return after computation.

#### See also

- [Batch](../../utils/datastruct.md#batch)

### Module().evaluate

[[find in source code]](blob/master/fz_openqa/modeling/modules/base.py#L174)

```python
def evaluate(batch: Batch, **kwargs):
```

Evaluate the model (step + step end) given a batch of data
with targets

#### See also

- [Batch](../../utils/datastruct.md#batch)

### Module().forward

[[find in source code]](blob/master/fz_openqa/modeling/modules/base.py#L168)

```python
def forward(batch: Batch, **kwargs):
```

Compute the forward pass of the model, does not require targets,
it can be used for inference.

#### See also

- [Batch](../../utils/datastruct.md#batch)

### Module().reset_metrics

[[find in source code]](blob/master/fz_openqa/modeling/modules/base.py#L267)

```python
def reset_metrics(split: Optional[Split] = None) -> None:
```

Reset the metrics corresponding to `split` if provided, else
reset all the metrics.

### Module().step

[[find in source code]](blob/master/fz_openqa/modeling/modules/base.py#L182)

```python
def step(batch: Batch, **kwargs: Any) -> Batch:
```

Compute the forward pass of the model and return output
Return a dictionary output with at least the key 'loss' and the data
necessary to compute the metrics, unless the loss is explicitly
computed in the `post_forward` method.

This step will be computed in the `*_step()` method of the
ligthning module: the data is processed separately on each device.

The torchmetric `Metric.update()` method should not be called here.
See `post_forward` instead.

Implement `_step` for each sub-class.

#### See also

- [Batch](../../utils/datastruct.md#batch)

### Module().step_end

[[find in source code]](blob/master/fz_openqa/modeling/modules/base.py#L200)

```python
def step_end(
    output: Batch,
    split: Optional[Split],
    update_metrics: bool = True,
    filter_features: bool = True,
) -> Any:
```

Apply a post-processing step to the forward method.
The output is the output of the forward method.

This method is called after the `output` has been gathered
from each device. This method must aggregate the loss across
devices.

torchmetrics update() calls should be placed here.
The output must at least contains the `loss` key.

Implement `_reduce_step_output` for each sub-class.

#### See also

- [Batch](../../utils/datastruct.md#batch)

### Module().update_metrics

[[find in source code]](blob/master/fz_openqa/modeling/modules/base.py#L262)

```python
def update_metrics(output: Batch, split: Split) -> None:
```

update the metrics of the given split.

#### See also

- [Batch](../../utils/datastruct.md#batch)

## init_default_heads

[[find in source code]](blob/master/fz_openqa/modeling/modules/base.py#L32)

```python
def init_default_heads():
```
