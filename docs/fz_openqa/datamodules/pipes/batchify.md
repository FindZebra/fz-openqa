# Batchify

> Auto-generated documentation for [fz_openqa.datamodules.pipes.batchify](blob/master/fz_openqa/datamodules/pipes/batchify.py) module.

- [Fz-openqa](../../../README.md#fz-openqa-index) / [Modules](../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../index.md#fz-openqa) / [Datamodules](../index.md#datamodules) / [Pipes](index.md#pipes) / Batchify
    - [AsBatch](#asbatch)
    - [Batchify](#batchify)
    - [DeBatchify](#debatchify)

## AsBatch

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/batchify.py#L24)

```python
class AsBatch(Pipe):
    def __init__(pipe: Pipe, **kwargs):
```

Apply a pipe to a single

## Batchify

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/batchify.py#L6)

```python
class Batchify(Pipe):
```

Convert an example into a batch

## DeBatchify

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/batchify.py#L14)

```python
class DeBatchify(Pipe):
```

Convert a one-element batch into am example
