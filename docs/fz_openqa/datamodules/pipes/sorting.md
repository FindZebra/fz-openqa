# Sorting

> Auto-generated documentation for [fz_openqa.datamodules.pipes.sorting](blob/master/fz_openqa/datamodules/pipes/sorting.py) module.

- [Fz-openqa](../../../README.md#fz-openqa-index) / [Modules](../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../index.md#fz-openqa) / [Datamodules](../index.md#datamodules) / [Pipes](index.md#pipes) / Sorting
    - [Sort](#sort)
    - [reindex](#reindex)

## Sort

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/sorting.py#L26)

```python
class Sort(Pipe):
    def __init__(keys: List[str], reverse: bool = True, **kwargs):
```

Sort a batch according to some key values

## reindex

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/sorting.py#L17)

```python
def reindex(x: Any, index: Union[np.ndarray, List[int]]) -> Any:
```
