# Json Struct

> Auto-generated documentation for [fz_openqa.utils.json_struct](blob/master/fz_openqa/utils/json_struct.py) module.

Function to process json-like structures

- [Fz-openqa](../../README.md#fz-openqa-index) / [Modules](../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../index.md#fz-openqa) / [Utils](index.md#utils) / Json Struct
    - [apply_to_json_struct](#apply_to_json_struct)
    - [flatten_json_struct](#flatten_json_struct)
    - [reduce_json_struct](#reduce_json_struct)

## apply_to_json_struct

[[find in source code]](blob/master/fz_openqa/utils/json_struct.py#L13)

```python
def apply_to_json_struct(
    data: Union[List, Dict],
    fn: Callable,
    **kwargs,
) -> Union[List, Dict]:
```

Apply a function to a json-like structure
Parameters
----------
data
    json-like structure
fn
    function to apply
kwargs
    keyword arguments to pass to fn

Returns
-------
json-like structure

## flatten_json_struct

[[find in source code]](blob/master/fz_openqa/utils/json_struct.py#L45)

```python
def flatten_json_struct(data: Union[List, Dict]) -> Iterable[Any]:
```

Flatten a json-like structure
Parameters
----------
data
    json-like structure
Yields
-------
Any
    Leaves of json-like structure

## reduce_json_struct

[[find in source code]](blob/master/fz_openqa/utils/json_struct.py#L69)

```python
def reduce_json_struct(
    data: Union[List, Dict],
    reduce_op: Callable[[Iterable[T]], T],
) -> T:
```

Reduce a json-like structure
Parameters
----------
data
    json-like structure
reduce_op
    reduce operation
Returns
-------
reduced json-like structure
