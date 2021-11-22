# SearchDocuments

> Auto-generated documentation for [fz_openqa.datamodules.pipelines.index.search_documents](blob/master/fz_openqa/datamodules/pipelines/index/search_documents.py) module.

- [Fz-openqa](../../../../README.md#fz-openqa-index) / [Modules](../../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../../index.md#fz-openqa) / [Datamodules](../../index.md#datamodules) / [Pipelines](../index.md#pipelines) / [Index](index.md#index) / SearchDocuments
    - [FetchNestedDocuments](#fetchnesteddocuments)
    - [SearchDocuments](#searchdocuments)

## FetchNestedDocuments

[[find in source code]](blob/master/fz_openqa/datamodules/pipelines/index/search_documents.py#L15)

```python
class FetchNestedDocuments(ApplyAsFlatten):
    def __init__(
        corpus_dataset: Dataset,
        collate_pipe: Pipe,
        update: bool = True,
        index_key: str = 'document.row_idx',
    ):
```

Retrieve the full document rows (text, input_ids, ...) from
the corpus object given the input `index_key` for nested documents ([[input_ids]])

## SearchDocuments

[[find in source code]](blob/master/fz_openqa/datamodules/pipelines/index/search_documents.py#L36)

```python
class SearchDocuments(Sequential):
    def __init__(corpus_index, n_documents: int):
```
