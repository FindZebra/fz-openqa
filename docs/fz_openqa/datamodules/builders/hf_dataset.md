# Hf Dataset

> Auto-generated documentation for [fz_openqa.datamodules.builders.hf_dataset](blob/master/fz_openqa/datamodules/builders/hf_dataset.py) module.

- [Fz-openqa](../../../README.md#fz-openqa-index) / [Modules](../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../index.md#fz-openqa) / [Datamodules](../index.md#datamodules) / [Builders](index.md#builders) / Hf Dataset
    - [HfDatasetBuilder](#hfdatasetbuilder)
        - [HfDatasetBuilder().filter_dataset](#hfdatasetbuilderfilter_dataset)
        - [HfDatasetBuilder().format_row](#hfdatasetbuilderformat_row)
        - [HfDatasetBuilder().get_collate_pipe](#hfdatasetbuilderget_collate_pipe)
        - [HfDatasetBuilder().load_and_filter_dataset](#hfdatasetbuilderload_and_filter_dataset)
        - [HfDatasetBuilder().load_base_dataset](#hfdatasetbuilderload_base_dataset)
        - [HfDatasetBuilder().preprocess_dataset](#hfdatasetbuilderpreprocess_dataset)
        - [HfDatasetBuilder().set_format](#hfdatasetbuilderset_format)
    - [cache_hf_dataset](#cache_hf_dataset)

## HfDatasetBuilder

[[find in source code]](blob/master/fz_openqa/datamodules/builders/hf_dataset.py#L66)

```python
class HfDatasetBuilder(DatasetBuilder):
    def __init__(
        tokenizer: PreTrainedTokenizerFast,
        add_encoding_tokens: bool = True,
        cache_dir: str = 'cache/',
        max_length: Optional[int] = 512,
        use_subset: bool = False,
        num_proc: int = 1,
        verbose: bool = False,
        text_formatter: Optional[TextFormatter] = None,
        **kwargs,
    ):
```

This class allows loading a preprocessing a `dataset.Dataset`

#### Attributes

- `dset_script_path_or_id` - HuggingFace dataset id or local path to script: `'ptb_text_only'`
- `text_field` - text field from the raw datasets that should be tokenized: `'sentence'`
- `pt_attributes` - name of the attributes that will be converted to
  tensors in the preprocessing function: `['input_ids', 'attention_mask']`
- `subset_size` - number of data points per subset train/val/test: `[100, 10, 10]`
- `column_names` - output columns: `['input_ids', 'attention_mask']`

#### See also

- [DatasetBuilder](base.md#datasetbuilder)

### HfDatasetBuilder().filter_dataset

[[find in source code]](blob/master/fz_openqa/datamodules/builders/hf_dataset.py#L136)

```python
def filter_dataset(dataset: HfDataset) -> HfDataset:
```

Apply filter operation to the dataset and return

#### See also

- [HfDataset](../utils/typing.md#hfdataset)

### HfDatasetBuilder().format_row

[[find in source code]](blob/master/fz_openqa/datamodules/builders/hf_dataset.py#L168)

```python
def format_row(row: Dict[str, Any]) -> str:
```

format a row from the dataset

### HfDatasetBuilder().get_collate_pipe

[[find in source code]](blob/master/fz_openqa/datamodules/builders/hf_dataset.py#L164)

```python
def get_collate_pipe() -> Pipe:
```

Returns a pipe that allow collating multiple rows into one Batch

### HfDatasetBuilder().load_and_filter_dataset

[[find in source code]](blob/master/fz_openqa/datamodules/builders/hf_dataset.py#L125)

```python
def load_and_filter_dataset() -> HfDataset:
```

#### See also

- [HfDataset](../utils/typing.md#hfdataset)

### HfDatasetBuilder().load_base_dataset

[[find in source code]](blob/master/fz_openqa/datamodules/builders/hf_dataset.py#L132)

```python
def load_base_dataset() -> DatasetDict:
```

Load the base HuggingFace dataset.

### HfDatasetBuilder().preprocess_dataset

[[find in source code]](blob/master/fz_openqa/datamodules/builders/hf_dataset.py#L140)

```python
def preprocess_dataset(dataset: HfDataset) -> HfDataset:
```

Apply processing steps to the dataset.
Tokenization and formatting as PyTorch tensors

#### See also

- [HfDataset](../utils/typing.md#hfdataset)

### HfDatasetBuilder().set_format

[[find in source code]](blob/master/fz_openqa/datamodules/builders/hf_dataset.py#L120)

```python
def set_format(dataset: HfDataset, format: str = 'torch') -> HfDataset:
```

#### See also

- [HfDataset](../utils/typing.md#hfdataset)

## cache_hf_dataset

[[find in source code]](blob/master/fz_openqa/datamodules/builders/hf_dataset.py#L26)

```python
def cache_hf_dataset(func):
```

Cache the output of the __call__ function by saving the dataset to a file.
Looks actually slower, but might be useful in a distributed setup.

Usage:

```
@cache_hf_dataset
def __call__(self, *args, **kwargs):
    ...
```
