# Pprint

> Auto-generated documentation for [fz_openqa.datamodules.pipes.pprint](blob/master/fz_openqa/datamodules/pipes/pprint.py) module.

- [Fz-openqa](../../../README.md#fz-openqa-index) / [Modules](../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../index.md#fz-openqa) / [Datamodules](../index.md#datamodules) / [Pipes](index.md#pipes) / Pprint
    - [PrintBatch](#printbatch)
    - [PrintText](#printtext)

## PrintBatch

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/pprint.py#L13)

```python
class PrintBatch(Pipe):
    def __init__(header: Optional[str] = None, **kwargs):
```

Print the batch

## PrintText

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/pprint.py#L40)

```python
class PrintText(Pipe):
    def __init__(
        text_key: str,
        limit: Optional[int] = None,
        header: Optional[str] = None,
        **kwargs,
    ):
```

Print the batch
