# Openqa

> Auto-generated documentation for [fz_openqa.datamodules.builders.openqa](blob/master/fz_openqa/datamodules/builders/openqa.py) module.

- [Fz-openqa](../../../README.md#fz-openqa-index) / [Modules](../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../index.md#fz-openqa) / [Datamodules](../index.md#datamodules) / [Builders](index.md#builders) / Openqa
    - [OpenQaBuilder](#openqabuilder)
        - [OpenQaBuilder().format_row](#openqabuilderformat_row)
        - [OpenQaBuilder().get_collate_pipe](#openqabuilderget_collate_pipe)
        - [OpenQaBuilder.get_select_documents_pipe](#openqabuilderget_select_documents_pipe)
        - [OpenQaBuilder().map_dataset](#openqabuildermap_dataset)
        - [OpenQaBuilder().pt_attributes](#openqabuilderpt_attributes)
        - [OpenQaBuilder().set_format](#openqabuilderset_format)
        - [OpenQaBuilder().to_dict](#openqabuilderto_dict)
    - [OpenQaDataset](#openqadataset)
        - [OpenQaDataset().new](#openqadatasetnew)

## OpenQaBuilder

[[find in source code]](blob/master/fz_openqa/datamodules/builders/openqa.py#L57)

```python
class OpenQaBuilder(DatasetBuilder):
    def __init__(
        dataset_builder: MedQaBuilder,
        corpus_builder: CorpusBuilder,
        index_builder: IndexBuilder,
        relevance_classifier: RelevanceClassifier,
        n_retrieved_documents: int,
        n_documents: Optional[Union[int, Dict]] = None,
        max_pos_docs: Optional[int] = None,
        filter_unmatched: bool = True,
        num_proc: int = 2,
        batch_size: int = 100,
        **kwargs,
    ):
```

#### See also

- [CorpusBuilder](corpus.md#corpusbuilder)
- [DatasetBuilder](base.md#datasetbuilder)
- [IndexBuilder](../index/builder.md#indexbuilder)
- [MedQaBuilder](medqa.md#medqabuilder)

### OpenQaBuilder().format_row

[[find in source code]](blob/master/fz_openqa/datamodules/builders/openqa.py#L272)

```python
def format_row(row: Dict[str, Any]) -> str:
```

### OpenQaBuilder().get_collate_pipe

[[find in source code]](blob/master/fz_openqa/datamodules/builders/openqa.py#L226)

```python
def get_collate_pipe() -> BlockSequential:
```

Build a Pipe to transform examples into a Batch.

### OpenQaBuilder.get_select_documents_pipe

[[find in source code]](blob/master/fz_openqa/datamodules/builders/openqa.py#L256)

```python
@staticmethod
def get_select_documents_pipe(
    n_documents: Union[int, Dict],
    max_pos_docs: Optional[int],
) -> Optional[Pipe]:
```

### OpenQaBuilder().map_dataset

[[find in source code]](blob/master/fz_openqa/datamodules/builders/openqa.py#L141)

```python
def map_dataset(
    dataset: DatasetDict,
    corpus: Dataset,
    index: Index,
    n_retrieved_documents: int,
    n_documents: int,
    max_pos_docs: Optional[int],
    num_proc: int,
    batch_size: int,
    relevance_classifier: RelevanceClassifier,
    filter_unmatched: bool,
) -> DatasetDict:
```

Map the dataset with documents from the corpus.

NB: SystemExit: 15: is due to an error in huggingface dataset when attempting
deleting the the dataset, see issue #114.

### OpenQaBuilder().pt_attributes

[[find in source code]](blob/master/fz_openqa/datamodules/builders/openqa.py#L105)

```python
@property
def pt_attributes():
```

### OpenQaBuilder().set_format

[[find in source code]](blob/master/fz_openqa/datamodules/builders/openqa.py#L136)

```python
def set_format(dataset: HfDataset, format: str = 'torch') -> HfDataset:
```

#### See also

- [HfDataset](../utils/typing.md#hfdataset)

### OpenQaBuilder().to_dict

[[find in source code]](blob/master/fz_openqa/datamodules/builders/openqa.py#L109)

```python
def to_dict() -> Dict[str, Any]:
```

## OpenQaDataset

[[find in source code]](blob/master/fz_openqa/datamodules/builders/openqa.py#L40)

```python
class OpenQaDataset(DatasetDict):
    def __init__(dataset: DatasetDict, corpus: Dataset, index: Index):
```

### OpenQaDataset().new

[[find in source code]](blob/master/fz_openqa/datamodules/builders/openqa.py#L46)

```python
def new(dataset: DatasetDict) -> 'OpenQaDataset':
```
