# Component

> Auto-generated documentation for [fz_openqa.datamodules.component](blob/master/fz_openqa/datamodules/component.py) module.

- [Fz-openqa](../../README.md#fz-openqa-index) / [Modules](../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../index.md#fz-openqa) / [Datamodules](index.md#datamodules) / Component
    - [Component](#component)
        - [Component().\_\_repr\_\_](#component__repr__)
        - [Component().copy](#componentcopy)
        - [Component().dill_inspect](#componentdill_inspect)
        - [Component().fingerprint](#componentfingerprint)
        - [Component().pprint](#componentpprint)
        - [Component.safe_fingerprint](#componentsafe_fingerprint)
        - [Component().to_json_struct](#componentto_json_struct)
    - [leaf_to_json_struct](#leaf_to_json_struct)

## Component

[[find in source code]](blob/master/fz_openqa/datamodules/component.py#L31)

```python
class Component():
    def __init__(id: Optional[str] = None, **kwargs):
```

A Component is an object used within the data processing pipeline.
Component implements a few method that helps integration with the rest of the framework,
such as safe serialization (pickle, multiprocessing) and deterministic caching (datasets).

Functionalities:
 - Serialization capability can be inspected using `dill_inspect()`
 - The hash/fingerprint of the object and its attributes can be obtained using `fingerprint()`
 - The object can be reduced to a json-struct using `to_json_struct()`
 - The object can be copied using `copy()`
 - The object can be printed using `pprint()`

----------
Attributes
id
   An identifier for the component.

### Component().\_\_repr\_\_

[[find in source code]](blob/master/fz_openqa/datamodules/component.py#L194)

```python
def __repr__() -> str:
```

Return a string representation of the object.

Returns
-------
str
    String representation of the object

### Component().copy

[[find in source code]](blob/master/fz_openqa/datamodules/component.py#L209)

```python
def copy(**kwargs):
```

Return a copy of the object and override the attributes using kwargs.

Parameters
----------
kwargs
    Attributes to override

Returns
-------
Component
    Copy of the object with overridden attributes

### Component().dill_inspect

[[find in source code]](blob/master/fz_openqa/datamodules/component.py#L66)

```python
def dill_inspect(reduce=True) -> Union[bool, Dict[str, bool]]:
```

Inspect the dill representation of the object.

Parameters
----------
reduce
    Collapse all the dill representations of the object into a single booleans

Returns
-------
Union[bool, Dict[str, Any]
    The dill representation of the object,
    allows both a boolean and a dictionary of booleans.
    If `reduce` is True, the dictionary is collapsed into a single boolean.

### Component().fingerprint

[[find in source code]](blob/master/fz_openqa/datamodules/component.py#L126)

```python
def fingerprint(reduce=False) -> Union[str, Dict[str, Any]]:
```

Return a fingerprint(s) of the object.

Returns
-------
Union[str, Dict[str, Any]]
    fingerprint(s) (hex-digested hash of the object),
    allows both a string and a nested structure of strings.

### Component().pprint

[[find in source code]](blob/master/fz_openqa/datamodules/component.py#L228)

```python
def pprint():
```

### Component.safe_fingerprint

[[find in source code]](blob/master/fz_openqa/datamodules/component.py#L119)

```python
@staticmethod
def safe_fingerprint(x: Any, reduce: bool = False) -> Union[Dict, str]:
```

### Component().to_json_struct

[[find in source code]](blob/master/fz_openqa/datamodules/component.py#L154)

```python
def to_json_struct(
    append_self: bool = False,
    exclude: Optional[List[str]] = None,
    **kwargs,
) -> Dict[str, Any]:
```

Return a dictionary representation of the object.

Returns
-------
Dictionary[str, Any]
    Dictionary representation of the object

## leaf_to_json_struct

[[find in source code]](blob/master/fz_openqa/datamodules/component.py#L19)

```python
def leaf_to_json_struct(v: Any, **kwargs) -> Union[Dict, List]:
```

Convert a leaf value into a json structure.
