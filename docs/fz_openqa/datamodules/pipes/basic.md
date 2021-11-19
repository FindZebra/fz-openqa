# Basic

> Auto-generated documentation for [fz_openqa.datamodules.pipes.basic](blob/master/fz_openqa/datamodules/pipes/basic.py) module.

- [Fz-openqa](../../../README.md#fz-openqa-index) / [Modules](../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../index.md#fz-openqa) / [Datamodules](../index.md#datamodules) / [Pipes](index.md#pipes) / Basic
    - [AddPrefix](#addprefix)
        - [AddPrefix().output_keys](#addprefixoutput_keys)
    - [Apply](#apply)
    - [ApplyToAll](#applytoall)
    - [CopyBatch](#copybatch)
    - [DropKeys](#dropkeys)
        - [DropKeys().output_keys](#dropkeysoutput_keys)
    - [FilterKeys](#filterkeys)
    - [GetKey](#getkey)
        - [GetKey().output_keys](#getkeyoutput_keys)
    - [Identity](#identity)
    - [Lambda](#lambda)
        - [Lambda().output_keys](#lambdaoutput_keys)
    - [RenameKeys](#renamekeys)
        - [RenameKeys().output_keys](#renamekeysoutput_keys)
    - [ReplaceInKeys](#replaceinkeys)
        - [ReplaceInKeys().output_keys](#replaceinkeysoutput_keys)

## AddPrefix

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/basic.py#L117)

```python
class AddPrefix(Pipe):
    def __init__(prefix: str, **kwargs):
```

Append the keys with a prefix.

### AddPrefix().output_keys

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/basic.py#L131)

```python
def output_keys(input_keys: List[str]) -> List[str]:
```

## Apply

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/basic.py#L197)

```python
class Apply(Pipe):
    def __init__(
        ops: Dict[str, Callable],
        element_wise: bool = False,
        **kwargs,
    ):
```

Transform the values in a batch using the transformations registered in `ops`
registered in `ops`: key, transformation`.
The argument `element_wise` allows to process each value in the batch element wise.

## ApplyToAll

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/basic.py#L235)

```python
class ApplyToAll(Pipe):
    def __init__(
        op: Callable,
        element_wise: bool = False,
        allow_kwargs: bool = False,
        **kwargs,
    ):
```

Apply a transformation
registered in `ops`: key, transformation`.
The argument `element_wise` allows to process each value in the batch element wise.

## CopyBatch

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/basic.py#L280)

```python
class CopyBatch(Pipe):
    def __init__(deep: bool = False, **kwargs):
```

Copy an input batch

## DropKeys

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/basic.py#L93)

```python
class DropKeys(Pipe):
    def __init__(keys: List[str], **kwargs):
```

Drop the keys in the current batch.

### DropKeys().output_keys

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/basic.py#L111)

```python
def output_keys(input_keys: List[str]) -> List[str]:
```

## FilterKeys

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/basic.py#L81)

```python
class FilterKeys(Identity):
    def __init__(condition: Optional[Condition], **kwargs):
```

Filter the keys in the batch given the `Condition` object.

#### See also

- [Identity](#identity)

## GetKey

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/basic.py#L64)

```python
class GetKey(Pipe):
    def __init__(key: str, **kwargs):
```

Returns a batch containing only the target key.

### GetKey().output_keys

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/basic.py#L77)

```python
def output_keys(input_keys: List[str]) -> List[str]:
```

## Identity

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/basic.py#L17)

```python
class Identity(Pipe):
```

A pipe that passes a batch without modifying it.

## Lambda

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/basic.py#L31)

```python
class Lambda(Pipe):
    def __init__(
        op: Callable,
        output_keys: Optional[List[str]] = None,
        allow_kwargs: bool = False,
        **kwargs,
    ):
```

Apply a lambda function to the batch.

### Lambda().output_keys

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/basic.py#L60)

```python
def output_keys(input_keys: List[str]) -> List[str]:
```

## RenameKeys

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/basic.py#L165)

```python
class RenameKeys(Pipe):
    def __init__(keys: Dict[str, str], **kwargs):
```

Rename a set of keys using a dictionary

### RenameKeys().output_keys

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/basic.py#L193)

```python
def output_keys(input_keys: List[str]) -> List[str]:
```

## ReplaceInKeys

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/basic.py#L135)

```python
class ReplaceInKeys(Pipe):
    def __init__(a: str, b: str, **kwargs):
```

Remove a pattern `a` with `b` in all keys

### ReplaceInKeys().output_keys

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/basic.py#L161)

```python
def output_keys(input_keys: List[str]) -> List[str]:
```
