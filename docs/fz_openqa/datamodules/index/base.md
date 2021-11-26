# Base

> Auto-generated documentation for [fz_openqa.datamodules.index.base](blob/master/fz_openqa/datamodules/index/base.py) module.

- [Fz-openqa](../../../README.md#fz-openqa-index) / [Modules](../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../index.md#fz-openqa) / [Datamodules](../index.md#datamodules) / [Index](index.md#index) / Base
    - [Index](#index)
        - [Index().build](#indexbuild)
        - [Index().get_example](#indexget_example)
        - [Index().search](#indexsearch)
        - [Index().search_one](#indexsearch_one)

## Index

[[find in source code]](blob/master/fz_openqa/datamodules/index/base.py#L17)

```python
class Index(Component):
    def __init__(dataset: Dataset, verbose: bool = False, **kwargs):
```

Keep an index of a Dataset and search u
sing queries.

#### See also

- [Component](../component.md#component)

### Index().build

[[find in source code]](blob/master/fz_openqa/datamodules/index/base.py#L36)

```python
@abstractmethod
def build(dataset: Dataset, **kwargs):
```

Index a dataset.

### Index().get_example

[[find in source code]](blob/master/fz_openqa/datamodules/index/base.py#L74)

```python
def get_example(query: Batch, index: int) -> Dict[str, Any]:
```

#### See also

- [Batch](../../utils/datastruct.md#batch)

### Index().search

[[find in source code]](blob/master/fz_openqa/datamodules/index/base.py#L47)

```python
def search(query: Batch, k: int = 1, **kwargs) -> SearchResult:
```

Batch search the index using the `query` and
return the scores and the indexes of the results
within the original dataset.

The default method search for each example sequentially.

#### See also

- [Batch](../../utils/datastruct.md#batch)
- [SearchResult](search_result.md#searchresult)

### Index().search_one

[[find in source code]](blob/master/fz_openqa/datamodules/index/base.py#L41)

```python
def search_one(
    query: Dict[str, Any],
    k: int = 1,
    **kwargs,
) -> Tuple[List[float], List[int], Optional[List[int]]]:
```

Search the index using one `query`
