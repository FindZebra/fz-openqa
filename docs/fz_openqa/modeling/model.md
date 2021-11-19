# Model

> Auto-generated documentation for [fz_openqa.modeling.model](blob/master/fz_openqa/modeling/model.py) module.

- [Fz-openqa](../../README.md#fz-openqa-index) / [Modules](../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../index.md#fz-openqa) / [Modeling](index.md#modeling) / Model
    - [Model](#model)
        - [Model().check_input_features](#modelcheck_input_features)
        - [Model().configure_optimizers](#modelconfigure_optimizers)
        - [Model().forward](#modelforward)
        - [Model().log_data](#modellog_data)
        - [Model().predict](#modelpredict)
        - [Model().test_epoch_end](#modeltest_epoch_end)
        - [Model().test_step](#modeltest_step)
        - [Model().test_step_end](#modeltest_step_end)
        - [Model().training_epoch_end](#modeltraining_epoch_end)
        - [Model().training_step](#modeltraining_step)
        - [Model().training_step_end](#modeltraining_step_end)
        - [Model().validation_epoch_end](#modelvalidation_epoch_end)
        - [Model().validation_step](#modelvalidation_step)
        - [Model().validation_step_end](#modelvalidation_step_end)

## Model

[[find in source code]](blob/master/fz_openqa/modeling/model.py#L22)

```python
class Model(LightningModule):
    def __init__(
        tokenizer: Union[PreTrainedTokenizerFast, DictConfig],
        bert: Union[BertPreTrainedModel, DictConfig],
        module: Union[DictConfig, Module],
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        **kwargs,
    ):
```

This class implements the basics of evaluation, logging and inference using
pytorch lightning mechanics. It contains a model: `nn.Module`.

## Main components
This class contains 2 main components:
* self.bert: the pretrained masked language model
* self.backbone: wraps the bert model is a specific head
* self.evaluator: handles computing the loss using the backbone and evaluate the metrics

## Pipeline
The main data processing flow can be described as follows:

    1.     batch = next(iter(dataloader))          (device=k)
                        |
        [   _step(batch): evaluator.step   ]    (processing on device k)
                        v
    2.             pre_output                      (device=k)
                        |
              [ gather (lightning) ]               (move data to device 0)
                        v
    3.              pre_output                     (device=0)
                        |
[ _step_end(pre_output): evaluator.step_end + log_data ]
                        v
    4.              output                         (device=0)

## Metrics:
The evaluator keeps track of the metrics using `torchmetrics`.
The metrics are updated at each `_step_end` (e.g. keeping track of
the true positives and false negatives).
The metrics are computed for the whole epoch in `_epoch_end`.

### Model().check_input_features

[[find in source code]](blob/master/fz_openqa/modeling/model.py#L222)

```python
def check_input_features(batch):
```

### Model().configure_optimizers

[[find in source code]](blob/master/fz_openqa/modeling/model.py#L167)

```python
def configure_optimizers():
```

Choose what optimizers and learning-rate schedulers to use in your optimization.
Normally you'd need one. But in the case of GANs or similar you might have multiple.
See examples here:
    https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    #configure-optimizers

### Model().forward

[[find in source code]](blob/master/fz_openqa/modeling/model.py#L92)

```python
def forward(batch: Batch, **kwargs) -> Batch:
```

#### See also

- [Batch](../utils/datastruct.md#batch)

### Model().log_data

[[find in source code]](blob/master/fz_openqa/modeling/model.py#L142)

```python
def log_data(
    data: Batch,
    prefix: Optional[str] = None,
    on_step=False,
    on_epoch=True,
    sync_dist=True,
):
```

Log all data from the input Batch. Only tensors with one elements are logged.
Each key is formatted as: `prefix/key` where prefix is usually
the split id.

#### See also

- [Batch](../utils/datastruct.md#batch)

### Model().predict

[[find in source code]](blob/master/fz_openqa/modeling/model.py#L95)

```python
def predict(batch: Batch, **kwargs) -> Batch:
```

#### See also

- [Batch](../utils/datastruct.md#batch)

### Model().test_epoch_end

[[find in source code]](blob/master/fz_openqa/modeling/model.py#L219)

```python
def test_epoch_end(outputs: List[Any]):
```

### Model().test_step

[[find in source code]](blob/master/fz_openqa/modeling/model.py#L205)

```python
def test_step(
    batch: Batch,
    batch_idx: int,
    dataloader_idx: Optional[int] = None,
) -> Batch:
```

#### See also

- [Batch](../utils/datastruct.md#batch)

### Model().test_step_end

[[find in source code]](blob/master/fz_openqa/modeling/model.py#L186)

```python
def test_step_end(batch: Batch, **kwargs) -> Batch:
```

#### See also

- [Batch](../utils/datastruct.md#batch)

### Model().training_epoch_end

[[find in source code]](blob/master/fz_openqa/modeling/model.py#L213)

```python
def training_epoch_end(outputs: List[Any]):
```

### Model().training_step

[[find in source code]](blob/master/fz_openqa/modeling/model.py#L189)

```python
def training_step(
    batch: Batch,
    batch_idx: int,
    dataloader_idx: Optional[int] = None,
) -> Batch:
```

#### See also

- [Batch](../utils/datastruct.md#batch)

### Model().training_step_end

[[find in source code]](blob/master/fz_openqa/modeling/model.py#L180)

```python
def training_step_end(batch: Batch, **kwargs) -> Batch:
```

#### See also

- [Batch](../utils/datastruct.md#batch)

### Model().validation_epoch_end

[[find in source code]](blob/master/fz_openqa/modeling/model.py#L216)

```python
def validation_epoch_end(outputs: List[Any]):
```

### Model().validation_step

[[find in source code]](blob/master/fz_openqa/modeling/model.py#L197)

```python
def validation_step(
    batch: Batch,
    batch_idx: int,
    dataloader_idx: Optional[int] = None,
) -> Batch:
```

#### See also

- [Batch](../utils/datastruct.md#batch)

### Model().validation_step_end

[[find in source code]](blob/master/fz_openqa/modeling/model.py#L183)

```python
def validation_step_end(batch: Batch, **kwargs) -> Batch:
```

#### See also

- [Batch](../utils/datastruct.md#batch)
