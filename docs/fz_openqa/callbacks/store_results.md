# Store Results

> Auto-generated documentation for [fz_openqa.callbacks.store_results](blob/master/fz_openqa/callbacks/store_results.py) module.

- [Fz-openqa](../../README.md#fz-openqa-index) / [Modules](../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../index.md#fz-openqa) / [Callbacks](index.md#callbacks) / Store Results
    - [StorePredictionsCallback](#storepredictionscallback)
        - [StorePredictionsCallback().is_written](#storepredictionscallbackis_written)
        - [StorePredictionsCallback().iter_batches](#storepredictionscallbackiter_batches)
        - [StorePredictionsCallback().on_predict_batch_end](#storepredictionscallbackon_predict_batch_end)
        - [StorePredictionsCallback().on_predict_epoch_end](#storepredictionscallbackon_predict_epoch_end)
        - [StorePredictionsCallback().on_predict_epoch_start](#storepredictionscallbackon_predict_epoch_start)
        - [StorePredictionsCallback().table](#storepredictionscallbacktable)

## StorePredictionsCallback

[[find in source code]](blob/master/fz_openqa/callbacks/store_results.py#L28)

```python
class StorePredictionsCallback(Callback):
    def __init__(
        cache_dir: Optional[str] = None,
        store_fields: Optional[List[str]] = None,
        cache_name: Optional[str] = None,
        persist: bool = True,
    ):
```

Allows storing the output of each `prediction_step` into a `pyarrow` table.
The Table can be access using the attribute `table` or using the method `iter_batches()`

### StorePredictionsCallback().is_written

[[find in source code]](blob/master/fz_openqa/callbacks/store_results.py#L99)

```python
@property
def is_written():
```

### StorePredictionsCallback().iter_batches

[[find in source code]](blob/master/fz_openqa/callbacks/store_results.py#L137)

```python
def iter_batches(batch_size=1000) -> Iterable[Batch]:
```

Returns an iterator over the cached batches

### StorePredictionsCallback().on_predict_batch_end

[[find in source code]](blob/master/fz_openqa/callbacks/store_results.py#L76)

```python
def on_predict_batch_end(
    trainer: 'pl.Trainer',
    pl_module: 'pl.LightningModule',
    outputs: Batch,
    batch: Batch,
    batch_idx: int,
    dataloader_idx: int,
) -> None:
```

store the outputs of the prediction step to the cache

#### See also

- [Batch](../utils/datastruct.md#batch)

### StorePredictionsCallback().on_predict_epoch_end

[[find in source code]](blob/master/fz_openqa/callbacks/store_results.py#L68)

```python
def on_predict_epoch_end(*args, **kwargs) -> None:
```

### StorePredictionsCallback().on_predict_epoch_start

[[find in source code]](blob/master/fz_openqa/callbacks/store_results.py#L71)

```python
def on_predict_epoch_start(
    trainer: 'pl.Trainer',
    pl_module: 'pl.LightningModule',
) -> None:
```

### StorePredictionsCallback().table

[[find in source code]](blob/master/fz_openqa/callbacks/store_results.py#L127)

```python
@property
def table() -> pa.Table:
```

Returns the cached `pyarrow.Table`
