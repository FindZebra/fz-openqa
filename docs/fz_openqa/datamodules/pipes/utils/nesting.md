# Nesting

> Auto-generated documentation for [fz_openqa.datamodules.pipes.utils.nesting](blob/master/fz_openqa/datamodules/pipes/utils/nesting.py) module.

- [Fz-openqa](../../../../README.md#fz-openqa-index) / [Modules](../../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../../index.md#fz-openqa) / [Datamodules](../../index.md#datamodules) / [Pipes](../index.md#pipes) / [Utils](index.md#utils) / Nesting
    - [flatten_nested](#flatten_nested)
    - [flatten\_nested\_](#flatten_nested_)
    - [nested_list](#nested_list)
    - [reconcat](#reconcat)

## flatten_nested

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/utils/nesting.py#L27)

```python
def flatten_nested(values: List[List], level=1) -> List[Any]:
```

Flatten a nested list of lists.

Parameters
----------
values
    The nested list to flatten.
level
    The level of nesting to flatten.
current_level
    The current level of nesting (don't set this manually).

Returns
-------
Iterable[Any]
    flatten list of values.

## flatten\_nested\_

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/utils/nesting.py#L15)

```python
def flatten_nested_(
    values: List[List],
    level=1,
    current_level=0,
) -> Iterable[Any]:
```

Flatten a nested list of lists. See [flatten_nested](#flatten_nested) for more details.

## nested_list

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/utils/nesting.py#L49)

```python
def nested_list(
    values: List[Any],
    shape: Union[Tuple[int], List[int]],
    level=0,
) -> List[Any]:
```

Nest a list of values according to `shape`. This functions is similar to `np.reshape`.

Parameters
----------
values
    The values to be nested.
shape
    The target shape
level
    Do not use. Used for debugging.

Returns
-------
List[Any]
    The nested list.

## reconcat

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/utils/nesting.py#L82)

```python
def reconcat(values: List[Any], original_type: Type):
```
