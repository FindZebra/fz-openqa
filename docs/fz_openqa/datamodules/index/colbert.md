# Colbert

> Auto-generated documentation for [fz_openqa.datamodules.index.colbert](blob/master/fz_openqa/datamodules/index/colbert.py) module.

- [Fz-openqa](../../../README.md#fz-openqa-index) / [Modules](../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../index.md#fz-openqa) / [Datamodules](../index.md#datamodules) / [Index](index.md#index) / Colbert
    - [ColbertIndex](#colbertindex)
        - [ColbertIndex().search](#colbertindexsearch)

## ColbertIndex

[[find in source code]](blob/master/fz_openqa/datamodules/index/colbert.py#L39)

```python
class ColbertIndex(FaissIndex):
    def __init__(dataset: Dataset, **kwargs):
```

### ColbertIndex().search

[[find in source code]](blob/master/fz_openqa/datamodules/index/colbert.py#L79)

```python
def search(query: Batch, k: int = 1, **kwargs) -> SearchResult:
```

Search the index using the `query` and
return the index of the results within the original dataset.
# todo: @idariis : this needs to be adapted for Colbert

#### See also

- [Batch](../../utils/datastruct.md#batch)
- [SearchResult](search_result.md#searchresult)
