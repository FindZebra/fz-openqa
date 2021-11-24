# Dataset

> Auto-generated documentation for [fz_openqa.datamodules.utils.dataset](blob/master/fz_openqa/datamodules/utils/dataset.py) module.

- [Fz-openqa](../../../README.md#fz-openqa-index) / [Modules](../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../index.md#fz-openqa) / [Datamodules](../index.md#datamodules) / [Utils](index.md#utils) / Dataset
    - [filter_questions_by_pos_docs](#filter_questions_by_pos_docs)
    - [format_size_difference](#format_size_difference)
    - [get_column_names](#get_column_names)
    - [take_subset](#take_subset)

## filter_questions_by_pos_docs

[[find in source code]](blob/master/fz_openqa/datamodules/utils/dataset.py#L47)

```python
def filter_questions_by_pos_docs(
    row,
    n_documents: Union[int, Dict],
    max_pos_docs: Optional[int],
    split: Optional[Split],
):
```

This function checks if a given row can should be filtered out.
It will be filtered out if
    1. There are no positive document.
    2. There are not enough negative documents to
       select `n_documents` with at max. `max_pos_docs` positive docs.

## format_size_difference

[[find in source code]](blob/master/fz_openqa/datamodules/utils/dataset.py#L36)

```python
def format_size_difference(
    original_size: Dict[str, int],
    new_dataset: DatasetDict,
) -> str:
```

## get_column_names

[[find in source code]](blob/master/fz_openqa/datamodules/utils/dataset.py#L16)

```python
def get_column_names(dataset: HfDataset) -> List[str]:
```

## take_subset

[[find in source code]](blob/master/fz_openqa/datamodules/utils/dataset.py#L23)

```python
def take_subset(dataset: HfDataset, subset_size: List[int]) -> HfDataset:
```

Take a subset of the dataset and return.
