# Documents

> Auto-generated documentation for [fz_openqa.datamodules.pipes.documents](blob/master/fz_openqa/datamodules/pipes/documents.py) module.

- [Fz-openqa](../../../README.md#fz-openqa-index) / [Modules](../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../index.md#fz-openqa) / [Datamodules](../index.md#datamodules) / [Pipes](index.md#pipes) / Documents
    - [SelectDocs](#selectdocs)
    - [SelectDocsOneEg](#selectdocsoneeg)
        - [SelectDocsOneEg().output_keys](#selectdocsoneegoutput_keys)
    - [select_values](#select_values)

## SelectDocs

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/documents.py#L20)

```python
class SelectDocs(Nested):
    def __init__(
        total: Union[int, Dict],
        max_pos_docs: Optional[int] = 1,
        pos_select_mode: str = 'first',
        neg_select_mode: str = 'first',
        strict: bool = False,
        prefix='document.',
        id='select-docs',
        **kwargs,
    ):
```

Select `total` documents with `max_pos_docs` positive documents (i.e. document.match_score>0)

## SelectDocsOneEg

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/documents.py#L46)

```python
class SelectDocsOneEg(Pipe):
    def __init__(
        total: Union[int, Dict],
        max_pos_docs: int = 1,
        pos_select_mode: str = 'first',
        neg_select_mode: str = 'first',
        strict: bool = True,
        **kwargs,
    ):
```

### SelectDocsOneEg().output_keys

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/documents.py#L64)

```python
def output_keys(input_keys: List[str]) -> List[str]:
```

## select_values

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/documents.py#L137)

```python
def select_values(
    values: List[int],
    k: int,
    mode: str = 'first',
) -> List[int]:
```
