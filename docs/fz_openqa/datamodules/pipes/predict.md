# Predict

> Auto-generated documentation for [fz_openqa.datamodules.pipes.predict](blob/master/fz_openqa/datamodules/pipes/predict.py) module.

- [Fz-openqa](../../../README.md#fz-openqa-index) / [Modules](../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../index.md#fz-openqa) / [Datamodules](../index.md#datamodules) / [Pipes](index.md#pipes) / Predict
    - [AddRowIdx](#addrowidx)
    - [Predict](#predict)
        - [Predict().cache](#predictcache)
        - [Predict.init_loader](#predictinit_loader)
        - [Predict().invalidate_cache](#predictinvalidate_cache)
        - [Predict().read_table](#predictread_table)
        - [Predict().to_json_struct](#predictto_json_struct)

## AddRowIdx

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/predict.py#L44)

```python
class AddRowIdx(TorchDataset):
    def __init__(dataset: Sized):
```

This class is used to add the column `IDX_COL` to the batch

## Predict

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/predict.py#L59)

```python
class Predict(Pipe):
    def __init__(
        model: pl.LightningModule,
        requires_cache: bool = False,
        **kwargs,
    ):
```

Allow computing predictions for a dataset using Lightning.
This pipes requires first to call `cache()` on the target dataset.
The predictions are then computed and stored to a cache file.
Once cached, the pipe can be called on the dataset again to
and the predictions are load from the cache

Notes
-----
This class can handle both Dataset and DatasetDict. However, the `idx` kwarg
must be passed to __call__ to successfully read the cache. This can be done using
the argument `with_indices` in `Dataset.map()` or `DatasetDict.map()`.

When using a `DatasetDict`, the kwarg `split` must be set to read the cache.

Attributes
----------
cache_file : Union[Path, str, tempfile.TemporaryFile]
    The cache file(s) to store the predictions.
    A dictionary is used when processing multiple splits.

Examples
---------
1. Using Predict on a Dataset:

```python
predict = Predict(model=model)

# cache the dataset: Dataset
predict.cache(dataset,
              trainer=trainer,
              collate_fn=collate_fn,
              cache_dir=cache_dir,
              loader_kwargs={'batch_size': 2},
              persist=True)

dataset = dataset.map(dataset,
                      batched=True,
                      batch_size=5,
                      with_indices=True)
```

2. Using Predict on a DatasetDict:

```python
predict = Predict(model=model)

# cache the dataset: DatasetDict
predict.cache(dataset,
              trainer=trainer,
              collate_fn=collate_fn,
              cache_dir=cache_dir,
              loader_kwargs={'batch_size': 2},
              persist=True)

dataset = DatasetDict({split: d.map(partial(predict, split=split),
                                    batched=True,
                                    batch_size=5,
                                    with_indices=True)
                       for split, d in dataset.items()})
```

### Predict().cache

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/predict.py#L214)

```python
@functools.singledispatchmethod
@torch.no_grad()
def cache(
    dataset: Dataset,
    trainer: Optional[Trainer] = None,
    collate_fn: Optional[Pipe] = None,
    loader_kwargs: Optional[Dict] = None,
    cache_dir: Optional[str] = None,
    split: Optional[Split] = None,
    persist: bool = True,
) -> CACHE_FILE:
```

Cache the predictions of the model on the dataset.

Parameters
----------
dataset
    The dataset to cache the predictions on.
trainer
    The pytorch_lightning Trainer used to accelerate the prediction
collate_fn
    The collate_fn used to process the dataset (e.g. builder.get_collate_pipe())
loader_kwargs
    The keyword arguments passed to the DataLoader
cache_dir
    The directory to store the cache file(s)
split
    (Optional) The split to cache the predictions on. Leave to None when using a Dataset.
persist
    (Optional) Whether to persist the cache file(s) for subsequent runs.
    If set to False, the cache file is deleted when the session ends (tempfile).
Returns
Union[Path, str, tempfile.TemporaryFile]
    The path to the cache file
-------

#### See also

- [CACHE_FILE](#cache_file)

### Predict.init_loader

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/predict.py#L365)

```python
@staticmethod
def init_loader(
    dataset: Dataset,
    collate_fn: Optional[Callable] = None,
    loader_kwargs: Optional[Dict] = None,
    wrap_indices: bool = True,
) -> DataLoader:
```

Initialize the dataloader.

if `wrap_indices` is True, the dataset and collate_fn are wrapped
such that each example comes with a valid `ROW_IDX` value.

Parameters
----------
dataset
    The dataset to initialize the dataloader for.
collate_fn
    The collate_fn to use for the dataloader.
loader_kwargs
    Additional keyword arguments to pass to the dataloader.
wrap_indices
    Whether to wrap the dataset and collate_fn with a valid `ROW_IDX` value.

Returns
-------
DataLoader
    The initialized dataloader.

### Predict().invalidate_cache

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/predict.py#L145)

```python
def invalidate_cache():
```

Reset the cache

### Predict().read_table

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/predict.py#L343)

```python
def read_table(split: Optional[Split]) -> pa.Table:
```

Returns the cached `pyarrow.Table` for the given split

### Predict().to_json_struct

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/predict.py#L448)

```python
def to_json_struct(
    exclude: Optional[List[str]] = None,
    **kwargs,
) -> Dict[str, Any]:
```

Override `to_json_struct` to exclude the cache location from the fingerprint.
# todo: unsure about the efficiency of this.
  if fingerprinting is not working as expected, this may be the cause.

Warnings
--------
exclude list will also be pass to the attribute pipes.
