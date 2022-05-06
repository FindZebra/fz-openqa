# Search

> Auto-generated documentation for [fz_openqa.datamodules.pipes.search](blob/master/fz_openqa/datamodules/pipes/search.py) module.

- [Fz-openqa](../../../README.md#fz-openqa-index) / [Modules](../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../index.md#fz-openqa) / [Datamodules](../index.md#datamodules) / [Pipes](index.md#pipes) / Search
    - [FakeDataset](#fakedataset)
        - [FakeDataset().\_\_getitem\_\_](#fakedataset__getitem__)
    - [FakeIndex](#fakeindex)
        - [FakeIndex().dill_inspect](#fakeindexdill_inspect)
        - [FakeIndex().search](#fakeindexsearch)
    - [FetchDocuments](#fetchdocuments)
        - [FetchDocuments().output_keys](#fetchdocumentsoutput_keys)
    - [SearchCorpus](#searchcorpus)
    - [SearchResult](#searchresult)

## FakeDataset

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/search.py#L43)

```python
class FakeDataset():
    def __init__():
```

A small class to test Search corpus without using a proper index

### FakeDataset().\_\_getitem\_\_

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/search.py#L49)

```python
def __getitem__(idx):
```

check if the module can be pickled.

## FakeIndex

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/search.py#L26)

```python
class FakeIndex():
```

A small class to test Search corpus without using a proper index

### FakeIndex().dill_inspect

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/search.py#L38)

```python
def dill_inspect() -> bool:
```

check if the module can be pickled.

### FakeIndex().search

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/search.py#L31)

```python
def search(query: Batch, k: int, **kwargs) -> SearchResult:
```

#### See also

- [Batch](../../utils/datastruct.md#batch)
- [SearchResult](#searchresult)

## FetchDocuments

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/search.py#L107)

```python
class FetchDocuments(Pipe):
    def __init__(
        corpus_dataset: Dataset,
        keys: Optional[List[str]] = None,
        collate_pipe: Pipe = None,
        index_key: str = 'document.row_idx',
        output_format: str = 'dict',
        id: str = 'fetch-documents-pipe',
        **kwargs,
    ):
```

### FetchDocuments().output_keys

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/search.py#L132)

```python
def output_keys(input_keys: List[str]) -> List[str]:
```

## SearchCorpus

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/search.py#L57)

```python
class SearchCorpus(Pipe):
    def __init__(
        corpus_index,
        k: Optional[int] = None,
        model: Optional[Union[Callable, torch.nn.Module]] = None,
        index_output_key: str = 'document.row_idx',
        score_output_key: str = 'document.proposal_score',
        analyzed_output_key: str = 'document.analyzed_tokens',
        **kwargs,
    ):
```

Search a Corpus object given a query

## SearchResult

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/search.py#L20)

```python
dataclass
class SearchResult():
```
