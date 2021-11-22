# Functional

> Auto-generated documentation for [fz_openqa.utils.functional](blob/master/fz_openqa/utils/functional.py) module.

- [Fz-openqa](../../README.md#fz-openqa-index) / [Modules](../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../index.md#fz-openqa) / [Utils](index.md#utils) / Functional
    - [always_true](#always_true)
    - [batch_reduce](#batch_reduce)
    - [cast_to_numpy](#cast_to_numpy)
    - [cast_values_to_numpy](#cast_values_to_numpy)
    - [check_equal_arrays](#check_equal_arrays)
    - [get_batch_eg](#get_batch_eg)
    - [infer_batch_size](#infer_batch_size)
    - [infer_device](#infer_device)
    - [infer_stride](#infer_stride)
    - [is_index_contiguous](#is_index_contiguous)
    - [is_loggable](#is_loggable)
    - [iter_batch_rows](#iter_batch_rows)
    - [maybe_instantiate](#maybe_instantiate)
    - [only_trainable](#only_trainable)

## always_true

[[find in source code]](blob/master/fz_openqa/utils/functional.py#L69)

```python
def always_true(*args, **kwargs):
```

## batch_reduce

[[find in source code]](blob/master/fz_openqa/utils/functional.py#L41)

```python
def batch_reduce(x, op=torch.sum):
```

## cast_to_numpy

[[find in source code]](blob/master/fz_openqa/utils/functional.py#L45)

```python
def cast_to_numpy(
    x: Any,
    as_contiguous: bool = True,
    dtype: Optional[np.dtype] = None,
) -> np.ndarray:
```

## cast_values_to_numpy

[[find in source code]](blob/master/fz_openqa/utils/functional.py#L63)

```python
def cast_values_to_numpy(
    batch: Batch,
    as_contiguous: bool = True,
    dtype: Optional[np.dtype] = None,
) -> Batch:
```

#### See also

- [Batch](datastruct.md#batch)

## check_equal_arrays

[[find in source code]](blob/master/fz_openqa/utils/functional.py#L101)

```python
def check_equal_arrays(x, y):
```

check if x==y

## get_batch_eg

[[find in source code]](blob/master/fz_openqa/utils/functional.py#L73)

```python
def get_batch_eg(
    batch: Batch,
    idx: int,
    filter_op: Optional[Callable] = None,
) -> Dict:
```

Extract example `idx` from a batch, potentially filter the keys

#### See also

- [Batch](datastruct.md#batch)

## infer_batch_size

[[find in source code]](blob/master/fz_openqa/utils/functional.py#L79)

```python
def infer_batch_size(batch: Batch) -> int:
```

#### See also

- [Batch](datastruct.md#batch)

## infer_device

[[find in source code]](blob/master/fz_openqa/utils/functional.py#L33)

```python
def infer_device(model):
```

## infer_stride

[[find in source code]](blob/master/fz_openqa/utils/functional.py#L91)

```python
def infer_stride(batch: Batch) -> int:
```

#### See also

- [Batch](datastruct.md#batch)

## is_index_contiguous

[[find in source code]](blob/master/fz_openqa/utils/functional.py#L115)

```python
def is_index_contiguous(indexes):
```

Check if indexes are contiguous: i.e I[i+1] = I[i] + 1

## is_loggable

[[find in source code]](blob/master/fz_openqa/utils/functional.py#L18)

```python
def is_loggable(x: Any):
```

## iter_batch_rows

[[find in source code]](blob/master/fz_openqa/utils/functional.py#L108)

```python
def iter_batch_rows(batch: Batch) -> Iterable[Dict]:
```

iterate through each batch example

#### See also

- [Batch](datastruct.md#batch)

## maybe_instantiate

[[find in source code]](blob/master/fz_openqa/utils/functional.py#L26)

```python
def maybe_instantiate(conf_or_obj: Union[Any, DictConfig], **kwargs):
```

## only_trainable

[[find in source code]](blob/master/fz_openqa/utils/functional.py#L37)

```python
def only_trainable(parameters: Iterable[torch.nn.Parameter]):
```
