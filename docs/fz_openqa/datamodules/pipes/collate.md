# Collate

> Auto-generated documentation for [fz_openqa.datamodules.pipes.collate](blob/master/fz_openqa/datamodules/pipes/collate.py) module.

- [Fz-openqa](../../../README.md#fz-openqa-index) / [Modules](../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../index.md#fz-openqa) / [Datamodules](../index.md#datamodules) / [Pipes](index.md#pipes) / Collate
    - [ApplyToEachExample](#applytoeachexample)
    - [Collate](#collate)
    - [DeCollate](#decollate)
    - [FirstEg](#firsteg)

## ApplyToEachExample

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/collate.py#L61)

```python
class ApplyToEachExample(Pipe):
    def __init__(pipe: Pipe, **kwargs):
```

## Collate

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/collate.py#L11)

```python
class Collate(Pipe):
    def __init__(keys: Optional[List[str]] = None, **kwargs):
```

Create a Batch object from a list of examples, where an
example is defined as a batch of one element.

This default class concatenate values as lists.

## DeCollate

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/collate.py#L39)

```python
class DeCollate(Pipe):
```

Returns a list of examples from a batch

## FirstEg

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/collate.py#L52)

```python
class FirstEg(Pipe):
```

Returns the first example
