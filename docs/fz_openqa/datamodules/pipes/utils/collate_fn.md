# Collate Fn

> Auto-generated documentation for [fz_openqa.datamodules.pipes.utils.collate_fn](blob/master/fz_openqa/datamodules/pipes/utils/collate_fn.py) module.

- [Fz-openqa](../../../../README.md#fz-openqa-index) / [Modules](../../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../../index.md#fz-openqa) / [Datamodules](../../index.md#datamodules) / [Pipes](../index.md#pipes) / [Utils](index.md#utils) / Collate Fn
    - [collate_and_pad_attributes](#collate_and_pad_attributes)
    - [collate_answer_options](#collate_answer_options)
    - [collate_nested_examples](#collate_nested_examples)
    - [collate_simple_attributes_by_key](#collate_simple_attributes_by_key)
    - [extract_and_collate_attributes_as_list](#extract_and_collate_attributes_as_list)

## collate_and_pad_attributes

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/utils/collate_fn.py#L25)

```python
def collate_and_pad_attributes(
    examples: List[Batch],
    tokenizer: PreTrainedTokenizerFast,
    key: Optional[str],
    exclude: Optional[Union[str, List[str]]] = None,
) -> Batch:
```

Collate the input encodings for a given key (e.g. "document", "question", ...)
using `PreTrainedTokenizerFast.pad`. Check the original documentation to see what types are
compatible. Return a Batch {'key.x' : ...., 'key.y': ....}

#### See also

- [Batch](../../../utils/datastruct.md#batch)

## collate_answer_options

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/utils/collate_fn.py#L127)

```python
def collate_answer_options(
    examples: List[Batch],
    tokenizer: PreTrainedTokenizerFast,
) -> Batch:
```

Collate the answer options, registered as separate fields ["answer_0.x", "answer_1.x", ...].
The return `answer_choices` tensor is of shape [batch_size, n_options, ...]

#### See also

- [Batch](../../../utils/datastruct.md#batch)

## collate_nested_examples

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/utils/collate_fn.py#L88)

```python
def collate_nested_examples(
    examples: List[List[Batch]],
    key: str,
    tokenizer: PreTrainedTokenizerFast,
):
```

Collate a list of list of examples, typically used when one
example features multiple documents.

## collate_simple_attributes_by_key

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/utils/collate_fn.py#L17)

```python
def collate_simple_attributes_by_key(examples, key: str, extract=False):
```

collate simple attributes such as the index

## extract_and_collate_attributes_as_list

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/utils/collate_fn.py#L60)

```python
def extract_and_collate_attributes_as_list(
    examples: List[Batch],
    attribute: str,
    key: Optional[str] = None,
) -> Tuple[List[Batch], Batch]:
```

Extract the attribute fields (e.g. `document.text`) from a list of Examples
and return all fields as a Batch `{'document.{attribute}': ["...", "..."]}`.
The target attributes are removed from the original examples
