# Es

> Auto-generated documentation for [fz_openqa.datamodules.index.es](blob/master/fz_openqa/datamodules/index/es.py) module.

- [Fz-openqa](../../../README.md#fz-openqa-index) / [Modules](../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../index.md#fz-openqa) / [Datamodules](../index.md#datamodules) / [Index](index.md#index) / Es
    - [ElasticSearchIndex](#elasticsearchindex)
        - [ElasticSearchIndex().build](#elasticsearchindexbuild)
        - [ElasticSearchIndex().preprocess_text](#elasticsearchindexpreprocess_text)
        - [ElasticSearchIndex().search](#elasticsearchindexsearch)
        - [ElasticSearchIndex().search_one](#elasticsearchindexsearch_one)

#### Attributes

- `DEFAULT_ES_BODY` - load the default es configuration: `OmegaConf.to_object(OmegaConf.load(Path(es_body...`

## ElasticSearchIndex

[[find in source code]](blob/master/fz_openqa/datamodules/index/es.py#L35)

```python
class ElasticSearchIndex(Index):
    def __init__(
        dataset: Dataset,
        index_key: str = 'document.row_idx',
        text_key: str = 'document.text',
        query_key: str = 'question.text',
        batch_size: int = 32,
        num_proc: int = 1,
        filter_mode: Optional[str] = None,
        text_cleaner: Optional[TextFormatter] = None,
        es_body: Optional[Dict] = DEFAULT_ES_BODY,
        analyze: Optional[bool] = False,
        **kwargs,
    ):
```

#### See also

- [DEFAULT_ES_BODY](#default_es_body)

### ElasticSearchIndex().build

[[find in source code]](blob/master/fz_openqa/datamodules/index/es.py#L92)

```python
def build(dataset: Dataset, verbose: bool = False, **kwargs):
```

Index the dataset using elastic search.
We make sure a unique index is created for each dataset

### ElasticSearchIndex().preprocess_text

[[find in source code]](blob/master/fz_openqa/datamodules/index/es.py#L175)

```python
def preprocess_text(dataset: Dataset) -> Dataset:
```

### ElasticSearchIndex().search

[[find in source code]](blob/master/fz_openqa/datamodules/index/es.py#L135)

```python
def search(query: Batch, k: int = 1, **kwargs) -> SearchResult:
```

Search the ES index for q batch of examples (query).

Filter the incoming batch using the same pipe as the one
used to build the index.

#### See also

- [Batch](../../utils/datastruct.md#batch)

### ElasticSearchIndex().search_one

[[find in source code]](blob/master/fz_openqa/datamodules/index/es.py#L157)

```python
def search_one(
    query: Dict[str, Any],
    field: str = None,
    k: int = 1,
    **kwargs,
) -> Tuple[List[float], List[int]]:
```

Search the index using the elastic search index for a single example.
