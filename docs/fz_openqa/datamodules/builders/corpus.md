# Corpus

> Auto-generated documentation for [fz_openqa.datamodules.builders.corpus](blob/master/fz_openqa/datamodules/builders/corpus.py) module.

- [Fz-openqa](../../../README.md#fz-openqa-index) / [Modules](../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../index.md#fz-openqa) / [Datamodules](../index.md#datamodules) / [Builders](index.md#builders) / Corpus
    - [CorpusBuilder](#corpusbuilder)
        - [CorpusBuilder().filter_dataset](#corpusbuilderfilter_dataset)
        - [CorpusBuilder().format_row](#corpusbuilderformat_row)
        - [CorpusBuilder().get_collate_pipe](#corpusbuilderget_collate_pipe)
        - [CorpusBuilder().get_generate_passages_pipe](#corpusbuilderget_generate_passages_pipe)
        - [CorpusBuilder().get_prefix_tokens](#corpusbuilderget_prefix_tokens)
        - [CorpusBuilder().get_tokenizer_pipe](#corpusbuilderget_tokenizer_pipe)
        - [CorpusBuilder().load_base_dataset](#corpusbuilderload_base_dataset)
        - [CorpusBuilder().preprocess_dataset](#corpusbuilderpreprocess_dataset)
    - [FZxMedQaCorpusBuilder](#fzxmedqacorpusbuilder)
        - [FZxMedQaCorpusBuilder().load_base_dataset](#fzxmedqacorpusbuilderload_base_dataset)
    - [FzCorpusBuilder](#fzcorpusbuilder)
    - [MedQaCorpusBuilder](#medqacorpusbuilder)
    - [WikipediaCorpusBuilder](#wikipediacorpusbuilder)

## CorpusBuilder

[[find in source code]](blob/master/fz_openqa/datamodules/builders/corpus.py#L44)

```python
class CorpusBuilder(HfDatasetBuilder):
    def __init__(
        passage_length: int = 200,
        passage_stride: int = 100,
        to_sentences: bool = False,
        input_dir: Optional[str] = None,
        append_document_title: bool = False,
        max_length: Optional[int] = None,
        **kwargs,
    ):
```

Builder for the Corpus Dataset.

Attributes
----------
dset_script_id
    HuggingFace dataset id or local path to script
dset_name
    Dataset name
pt_attributes
    name of the attributes that should be cast a Tensors
subset_size
    number of data points per subset train/val/test
output columns
    name of the columns

### CorpusBuilder().filter_dataset

[[find in source code]](blob/master/fz_openqa/datamodules/builders/corpus.py#L136)

```python
def filter_dataset(dataset: HfDataset) -> HfDataset:
```

Apply filter operation to the dataset and return

#### See also

- [HfDataset](../utils/typing.md#hfdataset)

### CorpusBuilder().format_row

[[find in source code]](blob/master/fz_openqa/datamodules/builders/corpus.py#L261)

```python
def format_row(row: Dict[str, Any]) -> str:
```

Decode and print one example from the batch

### CorpusBuilder().get_collate_pipe

[[find in source code]](blob/master/fz_openqa/datamodules/builders/corpus.py#L240)

```python
def get_collate_pipe() -> Pipe:
```

Build a Pipe to transform examples into a Batch.

### CorpusBuilder().get_generate_passages_pipe

[[find in source code]](blob/master/fz_openqa/datamodules/builders/corpus.py#L198)

```python
def get_generate_passages_pipe():
```

Build the pipe to extract overlapping passages from the tokenized documents.

### CorpusBuilder().get_prefix_tokens

[[find in source code]](blob/master/fz_openqa/datamodules/builders/corpus.py#L232)

```python
def get_prefix_tokens():
```

### CorpusBuilder().get_tokenizer_pipe

[[find in source code]](blob/master/fz_openqa/datamodules/builders/corpus.py#L209)

```python
def get_tokenizer_pipe():
```

Build a pipe to tokenize raw documents, a shortcut with the Pipe
Parallel is added to return the original attributes as well.

### CorpusBuilder().load_base_dataset

[[find in source code]](blob/master/fz_openqa/datamodules/builders/corpus.py#L118)

```python
def load_base_dataset() -> DatasetDict:
```

Load the base HuggingFace dataset.

### CorpusBuilder().preprocess_dataset

[[find in source code]](blob/master/fz_openqa/datamodules/builders/corpus.py#L140)

```python
def preprocess_dataset(dataset: HfDataset) -> HfDataset:
```

Apply processing steps to the dataset. Tokenization and formatting as PyTorch tensors

#### See also

- [HfDataset](../utils/typing.md#hfdataset)

## FZxMedQaCorpusBuilder

[[find in source code]](blob/master/fz_openqa/datamodules/builders/corpus.py#L281)

```python
class FZxMedQaCorpusBuilder(CorpusBuilder):
```

#### See also

- [CorpusBuilder](#corpusbuilder)

### FZxMedQaCorpusBuilder().load_base_dataset

[[find in source code]](blob/master/fz_openqa/datamodules/builders/corpus.py#L288)

```python
def load_base_dataset() -> DatasetDict:
```

## FzCorpusBuilder

[[find in source code]](blob/master/fz_openqa/datamodules/builders/corpus.py#L276)

```python
class FzCorpusBuilder(CorpusBuilder):
```

#### See also

- [CorpusBuilder](#corpusbuilder)

## MedQaCorpusBuilder

[[find in source code]](blob/master/fz_openqa/datamodules/builders/corpus.py#L271)

```python
class MedQaCorpusBuilder(CorpusBuilder):
```

#### See also

- [CorpusBuilder](#corpusbuilder)

## WikipediaCorpusBuilder

[[find in source code]](blob/master/fz_openqa/datamodules/builders/corpus.py#L295)

```python
class WikipediaCorpusBuilder(CorpusBuilder):
```

#### See also

- [CorpusBuilder](#corpusbuilder)
