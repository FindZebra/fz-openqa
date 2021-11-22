# Dense

> Auto-generated documentation for [fz_openqa.datamodules.index.dense](blob/master/fz_openqa/datamodules/index/dense.py) module.

- [Fz-openqa](../../../README.md#fz-openqa-index) / [Modules](../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../index.md#fz-openqa) / [Datamodules](../index.md#datamodules) / [Index](index.md#index) / Dense
    - [FaissIndex](#faissindex)
- [todo: handle temporary cache_dir form here (persist=False)](#todo-handle-temporary-cache_dir-form-here-persistfalse)
        - [FaissIndex().build](#faissindexbuild)
        - [FaissIndex().cache_query_dataset](#faissindexcache_query_dataset)
        - [FaissIndex().get_rename_output_names_pipe](#faissindexget_rename_output_names_pipe)
        - [FaissIndex().is_indexed](#faissindexis_indexed)
        - [FaissIndex().search](#faissindexsearch)
    - [iter_batches_with_indexes](#iter_batches_with_indexes)

## FaissIndex

[[find in source code]](blob/master/fz_openqa/datamodules/index/dense.py#L65)

```python
class FaissIndex(Index):
    def __init__(
        dataset: Dataset,
        model: pl.LightningModule,
        trainer: Optional[Trainer] = None,
        faiss_args: Dict[str, Any] = None,
        index_key: str = 'document.row_idx',
        model_output_keys: List[str],
        loader_kwargs: Optional[Dict] = None,
        collate_pipe: Pipe = None,
        persist_cache: bool = True,
        cache_dir: Optional[str] = None,
        **kwargs,
    ):
```

A dense index using Faiss. This class allows storing document vectors (one-dimensional)
into a Faiss index and allows querying the index for similar documents.

This class allows processing `Dataset` using  a `pl.LighningModule` and
(optionally) a `pl.Trainer`. This is done automatically when passing a Trainer in the
constructor. This functionality relies on the `Predict` pipe, which is used to
handle PyTorch Lightning mechanics. `Predict` works by caching predictions to a
`pyarrow.Table`. Indexes must be passed (i.e. `pipe(batch, idx=idx)`) at all times to
enables the `Predict` pipe to work.

The trainer can also be used to process the queries. In that case either
1. call `cache_query_dataset` before calling `search` on each batch`
2. call `search` directly on the query `Dataset`

# todo: handle temporary cache_dir form here (persist=False)

Parameters
----------
model
    The model to use for indexing and querying, e.g. a `pl.LightningModule`.
index_name
    The name of the index. Must be unique for each configuration.

#### See also

- [Index](base.md#index)

### FaissIndex().build

[[find in source code]](blob/master/fz_openqa/datamodules/index/dense.py#L191)

```python
def build(
    dataset: Dataset,
    name: Optional[str] = None,
    cache_dir=Optional[None],
    **kwargs,
):
```

Build and cache the index. Cache is skipped if `name` or `cache_dir` is not provided.

Parameters
----------
dataset
    The dataset to index.
name
    The name of the index (must be unique).
cache_dir
    The directory to store the cache in.
kwargs
    Additional arguments to pass to `_build`
Returns
-------
None

### FaissIndex().cache_query_dataset

[[find in source code]](blob/master/fz_openqa/datamodules/index/dense.py#L409)

```python
def cache_query_dataset(
    dataset: Union[Dataset, DatasetDict],
    collate_fn: Callable,
    **kwargs,
):
```

### FaissIndex().get_rename_output_names_pipe

[[find in source code]](blob/master/fz_openqa/datamodules/index/dense.py#L466)

```python
def get_rename_output_names_pipe(inputs: List[str], output: str) -> Pipe:
```

Format the output of the model

### FaissIndex().is_indexed

[[find in source code]](blob/master/fz_openqa/datamodules/index/dense.py#L187)

```python
@property
def is_indexed():
```

### FaissIndex().search

[[find in source code]](blob/master/fz_openqa/datamodules/index/dense.py#L333)

```python
@singledispatchmethod
def search(
    query: Batch,
    idx: Optional[List[int]] = None,
    split: Optional[Split] = None,
    k: int = 1,
    **kwargs,
) -> SearchResult:
```

Search the index using a batch of queries. For a single query, the batch is processed
using the model and the predict pipe.
For a whole query dataset, you might consider calling [FaissIndex().cache_query_dataset](#faissindexcache_query_dataset) first. In that
case you must provide the `idx` and `split` arguments (see Warnings).
This is however handled automatically when calling `search` with a `Dataset` as query.

Warnings
--------
`idx` and `split` must be provided if the queried dataset was cached. Caching is
performed if calling `cache_query_dataset()` beforehand.

Parameters
----------
query
    The batch of queries to search for.
idx
    The indexes to search in.
split
    The dataset split to search in.
k
kwargs

Returns
-------
SearchResult
    The search result for the batch

#### See also

- [Batch](../../utils/datastruct.md#batch)
- [SearchResult](search_result.md#searchresult)

## iter_batches_with_indexes

[[find in source code]](blob/master/fz_openqa/datamodules/index/dense.py#L47)

```python
def iter_batches_with_indexes(
    loader: Union[Generator, DataLoader],
) -> Iterable[Tuple[List[int], Batch]]:
```

Iterate over batches and return a tuple of the batch index and the batch itself.
