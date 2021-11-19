# Utils

> Auto-generated documentation for [fz_openqa.modeling.modules.utils](blob/master/fz_openqa/modeling/modules/utils.py) module.

- [Fz-openqa](../../../README.md#fz-openqa-index) / [Modules](../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../index.md#fz-openqa) / [Modeling](../index.md#modeling) / [Modules](index.md#modules) / Utils
    - [check_only_first_doc_positive](#check_only_first_doc_positive)
    - [expand_and_flatten](#expand_and_flatten)
    - [flatten_first_dims](#flatten_first_dims)

## check_only_first_doc_positive

[[find in source code]](blob/master/fz_openqa/modeling/modules/utils.py#L8)

```python
def check_only_first_doc_positive(batch, match_key='document.match_score'):
```

## expand_and_flatten

[[find in source code]](blob/master/fz_openqa/modeling/modules/utils.py#L16)

```python
def expand_and_flatten(batch: Batch, n_docs, keys: List[str]) -> Batch:
```

#### See also

- [Batch](../../utils/datastruct.md#batch)

## flatten_first_dims

[[find in source code]](blob/master/fz_openqa/modeling/modules/utils.py#L25)

```python
def flatten_first_dims(batch: Batch, n_dims, keys: List[str]) -> Batch:
```

Collapse the first `n_dims` into a single dimension.

#### See also

- [Batch](../../utils/datastruct.md#batch)
