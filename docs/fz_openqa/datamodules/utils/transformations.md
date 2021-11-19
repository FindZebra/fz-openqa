# Transformations

> Auto-generated documentation for [fz_openqa.datamodules.utils.transformations](blob/master/fz_openqa/datamodules/utils/transformations.py) module.

- [Fz-openqa](../../../README.md#fz-openqa-index) / [Modules](../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../index.md#fz-openqa) / [Datamodules](../index.md#datamodules) / [Utils](index.md#utils) / Transformations
    - [add_spec_token](#add_spec_token)
    - [append_document_title](#append_document_title)
    - [set_row_idx](#set_row_idx)
    - [truncate_examples_to_max_length](#truncate_examples_to_max_length)

## add_spec_token

[[find in source code]](blob/master/fz_openqa/datamodules/utils/transformations.py#L12)

```python
def add_spec_token(special_token: str, text: str):
```

This functions append a special token to a text such that output = special_token+text.
The pretrained tokenizer with registered special tokens will encode the output as:
[CLS][SPEC][ text tokens ][SEP]

## append_document_title

[[find in source code]](blob/master/fz_openqa/datamodules/utils/transformations.py#L29)

```python
def append_document_title(example: Dict[str, Any]) -> Dict[str, Any]:
```

## set_row_idx

[[find in source code]](blob/master/fz_openqa/datamodules/utils/transformations.py#L25)

```python
def set_row_idx(
    example: Dict[str, Any],
    idx: int,
    key: str = 'idx',
) -> Dict[str, Any]:
```

## truncate_examples_to_max_length

[[find in source code]](blob/master/fz_openqa/datamodules/utils/transformations.py#L34)

```python
def truncate_examples_to_max_length(
    output,
    key: str,
    tokenizer: PreTrainedTokenizerFast,
):
```
