# Medqa

> Auto-generated documentation for [fz_openqa.datamodules.builders.medqa](blob/master/fz_openqa/datamodules/builders/medqa.py) module.

- [Fz-openqa](../../../README.md#fz-openqa-index) / [Modules](../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../index.md#fz-openqa) / [Datamodules](../index.md#datamodules) / [Builders](index.md#builders) / Medqa
    - [MedQaBuilder](#medqabuilder)
        - [MedQaBuilder().filter_dataset](#medqabuilderfilter_dataset)
        - [MedQaBuilder().format_row](#medqabuilderformat_row)
        - [MedQaBuilder().get_answer_tokenizer_pipe](#medqabuilderget_answer_tokenizer_pipe)
        - [MedQaBuilder().get_collate_pipe](#medqabuilderget_collate_pipe)
        - [MedQaBuilder().get_question_tokenizer_pipe](#medqabuilderget_question_tokenizer_pipe)
        - [MedQaBuilder().load_base_dataset](#medqabuilderload_base_dataset)
        - [MedQaBuilder().preprocess_dataset](#medqabuilderpreprocess_dataset)

## MedQaBuilder

[[find in source code]](blob/master/fz_openqa/datamodules/builders/medqa.py#L29)

```python
class MedQaBuilder(HfDatasetBuilder):
```

#### Attributes

- `dset_script_path_or_id` - HuggingFace dataset id or local path to script: `medqa.__file__`
- `pt_attributes` - name of the attributes that will be converted to
  tensors in the preprocessing function: `['question.input_ids', 'question.attention_mask...`
- `subset_size` - number of data points per subset train/val/test: `[100, 50, 50]`
- `n_options` - number of options: `4`
- `column_names` - output columns: `['answer.text', 'answer.input_ids', 'answer.att...`

### MedQaBuilder().filter_dataset

[[find in source code]](blob/master/fz_openqa/datamodules/builders/medqa.py#L67)

```python
def filter_dataset(dataset: HfDataset) -> HfDataset:
```

Apply filter operation to the dataset and return

#### See also

- [HfDataset](../utils/typing.md#hfdataset)

### MedQaBuilder().format_row

[[find in source code]](blob/master/fz_openqa/datamodules/builders/medqa.py#L133)

```python
def format_row(row: Dict[str, Any]) -> str:
```

Decode and print one row from the batch

### MedQaBuilder().get_answer_tokenizer_pipe

[[find in source code]](blob/master/fz_openqa/datamodules/builders/medqa.py#L97)

```python
def get_answer_tokenizer_pipe():
```

### MedQaBuilder().get_collate_pipe

[[find in source code]](blob/master/fz_openqa/datamodules/builders/medqa.py#L120)

```python
def get_collate_pipe():
```

### MedQaBuilder().get_question_tokenizer_pipe

[[find in source code]](blob/master/fz_openqa/datamodules/builders/medqa.py#L108)

```python
def get_question_tokenizer_pipe():
```

create a Pipe to tokenize the questions.

### MedQaBuilder().load_base_dataset

[[find in source code]](blob/master/fz_openqa/datamodules/builders/medqa.py#L63)

```python
def load_base_dataset() -> DatasetDict:
```

Load the base HuggingFace dataset.

### MedQaBuilder().preprocess_dataset

[[find in source code]](blob/master/fz_openqa/datamodules/builders/medqa.py#L71)

```python
def preprocess_dataset(dataset: HfDataset) -> HfDataset:
```

Apply processing steps to the dataset.
Tokenization and formatting as PyTorch tensors

#### See also

- [HfDataset](../utils/typing.md#hfdataset)
