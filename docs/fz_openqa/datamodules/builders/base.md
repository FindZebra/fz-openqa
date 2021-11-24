# Base

> Auto-generated documentation for [fz_openqa.datamodules.builders.base](blob/master/fz_openqa/datamodules/builders/base.py) module.

- [Fz-openqa](../../../README.md#fz-openqa-index) / [Modules](../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../index.md#fz-openqa) / [Datamodules](../index.md#datamodules) / [Builders](index.md#builders) / Base
    - [DatasetBuilder](#datasetbuilder)
        - [DatasetBuilder().format_row](#datasetbuilderformat_row)
        - [DatasetBuilder().get_collate_pipe](#datasetbuilderget_collate_pipe)
    - [to_snake_format](#to_snake_format)

## DatasetBuilder

[[find in source code]](blob/master/fz_openqa/datamodules/builders/base.py#L23)

```python
class DatasetBuilder():
    def __init__(cache_dir: Optional[str]):
```

DatasetBuilder is a class that is responsible for building a dataset.

### DatasetBuilder().format_row

[[find in source code]](blob/master/fz_openqa/datamodules/builders/base.py#L48)

```python
def format_row(row: Dict[str, Any]) -> str:
```

format a row from the dataset

### DatasetBuilder().get_collate_pipe

[[find in source code]](blob/master/fz_openqa/datamodules/builders/base.py#L45)

```python
def get_collate_pipe() -> Pipe:
```

## to_snake_format

[[find in source code]](blob/master/fz_openqa/datamodules/builders/base.py#L18)

```python
def to_snake_format(name: str) -> str:
```

convert a class name (Camel style) to file style (Snake style)
