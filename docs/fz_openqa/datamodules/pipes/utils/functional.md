# Functional

> Auto-generated documentation for [fz_openqa.datamodules.pipes.utils.functional](blob/master/fz_openqa/datamodules/pipes/utils/functional.py) module.

- [Fz-openqa](../../../../README.md#fz-openqa-index) / [Modules](../../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../../index.md#fz-openqa) / [Datamodules](../../index.md#datamodules) / [Pipes](../index.md#pipes) / [Utils](index.md#utils) / Functional
    - [reduce_dict_values](#reduce_dict_values)
    - [safe_fingerprint](#safe_fingerprint)
    - [safe_todict](#safe_todict)

## reduce_dict_values

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/utils/functional.py#L28)

```python
def reduce_dict_values(x: Union[bool, Dict[str, Any]], op=all) -> bool:
```

Reduce a nested dictionary structure with boolean values
into a single boolean output.

## safe_fingerprint

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/utils/functional.py#L20)

```python
def safe_fingerprint(x):
```

Return the fingerprint of `x`, even if `x` is not a `Pipe`.

## safe_todict

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/utils/functional.py#L9)

```python
def safe_todict(x):
```

return a dictionary representation of `x`, even if `x` is not a `Pipe`.
