# SearchResult

> Auto-generated documentation for [fz_openqa.datamodules.index.search_result](blob/master/fz_openqa/datamodules/index/search_result.py) module.

- [Fz-openqa](../../../README.md#fz-openqa-index) / [Modules](../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../index.md#fz-openqa) / [Datamodules](../index.md#datamodules) / [Index](index.md#index) / SearchResult
    - [SearchResult](#searchresult)

## SearchResult

[[find in source code]](blob/master/fz_openqa/datamodules/index/search_result.py#L9)

```python
class SearchResult():
    def __init__(
        score: List[List[float]],
        index: List[List[int]],
        tokens: Optional[List[List[str]]] = None,
        dataset_size: Optional[int] = None,
    ):
```

A small class to help handling the search results.
