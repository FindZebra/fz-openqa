# Es Engine

> Auto-generated documentation for [fz_openqa.datamodules.index.utils.es_engine](blob/master/fz_openqa/datamodules/index/utils/es_engine.py) module.

- [Fz-openqa](../../../../README.md#fz-openqa-index) / [Modules](../../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../../index.md#fz-openqa) / [Datamodules](../../index.md#datamodules) / [Index](../index.md#index) / [Utils](index.md#utils) / Es Engine
    - [ElasticSearchEngine](#elasticsearchengine)
        - [ElasticSearchEngine().\_\_getstate\_\_](#elasticsearchengine__getstate__)
        - [ElasticSearchEngine().es_analyze_text](#elasticsearchenginees_analyze_text)
        - [ElasticSearchEngine().es_bulk](#elasticsearchenginees_bulk)
        - [ElasticSearchEngine().es_create_index](#elasticsearchenginees_create_index)
        - [ElasticSearchEngine().es_ingest](#elasticsearchenginees_ingest)
        - [ElasticSearchEngine().es_remove_index](#elasticsearchenginees_remove_index)
        - [ElasticSearchEngine().es_search](#elasticsearchenginees_search)
        - [ElasticSearchEngine().es_search_bulk](#elasticsearchenginees_search_bulk)
        - [ElasticSearchEngine().instance](#elasticsearchengineinstance)
    - [ping_es](#ping_es)

## ElasticSearchEngine

[[find in source code]](blob/master/fz_openqa/datamodules/index/utils/es_engine.py#L19)

```python
class ElasticSearchEngine():
    def __init__(timeout=60, analyze=False):
```

### ElasticSearchEngine().\_\_getstate\_\_

[[find in source code]](blob/master/fz_openqa/datamodules/index/utils/es_engine.py#L32)

```python
def __getstate__():
```

this method is called when attempting pickling.
ES instances cannot be properly pickled

### ElasticSearchEngine().es_analyze_text

[[find in source code]](blob/master/fz_openqa/datamodules/index/utils/es_engine.py#L160)

```python
def es_analyze_text(index_name: str, queries: List[str]):
```

### ElasticSearchEngine().es_bulk

[[find in source code]](blob/master/fz_openqa/datamodules/index/utils/es_engine.py#L85)

```python
def es_bulk(
    index_name: str,
    title: str,
    document_idx: list,
    document_txt: list,
):
```

### ElasticSearchEngine().es_create_index

[[find in source code]](blob/master/fz_openqa/datamodules/index/utils/es_engine.py#L54)

```python
def es_create_index(index_name: str, body: Optional[Dict] = None) -> bool:
```

Create ElasticSearch Index

### ElasticSearchEngine().es_ingest

[[find in source code]](blob/master/fz_openqa/datamodules/index/utils/es_engine.py#L78)

```python
def es_ingest(index_name: str, title: str, paragraph: str):
```

Ingest to ElasticSearch Index

### ElasticSearchEngine().es_remove_index

[[find in source code]](blob/master/fz_openqa/datamodules/index/utils/es_engine.py#L72)

```python
def es_remove_index(index_name: str):
```

Remove ElasticSearch Index

### ElasticSearchEngine().es_search

[[find in source code]](blob/master/fz_openqa/datamodules/index/utils/es_engine.py#L145)

```python
def es_search(index_name: str, query: str, results: int):
```

Sequential search in ElasticSearch Index

### ElasticSearchEngine().es_search_bulk

[[find in source code]](blob/master/fz_openqa/datamodules/index/utils/es_engine.py#L113)

```python
def es_search_bulk(index_name: str, queries: List[str], k: int):
```

Batch search in ElasticSearch Index

### ElasticSearchEngine().instance

[[find in source code]](blob/master/fz_openqa/datamodules/index/utils/es_engine.py#L50)

```python
@property
def instance():
```

## ping_es

[[find in source code]](blob/master/fz_openqa/datamodules/index/utils/es_engine.py#L15)

```python
def ping_es() -> bool:
```
