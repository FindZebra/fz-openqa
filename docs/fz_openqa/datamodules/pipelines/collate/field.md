# Field

> Auto-generated documentation for [fz_openqa.datamodules.pipelines.collate.field](blob/master/fz_openqa/datamodules/pipelines/collate/field.py) module.

- [Fz-openqa](../../../../README.md#fz-openqa-index) / [Modules](../../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../../index.md#fz-openqa) / [Datamodules](../../index.md#datamodules) / [Pipelines](../index.md#pipelines) / [Collate](index.md#collate) / Field
    - [CollateField](#collatefield)
    - [to_tensor_op](#to_tensor_op)

## CollateField

[[find in source code]](blob/master/fz_openqa/datamodules/pipelines/collate/field.py#L36)

```python
class CollateField(Gate):
    def __init__(
        field: str,
        tokenizer: Optional[PreTrainedTokenizerFast] = None,
        exclude: Optional[List[str]] = None,
        to_tensor: Optional[List[str]] = None,
        level: int = 0,
        **kwargs,
    ):
```

Collate examples for a given field.
Field corresponds to the prefix of the keys (field.attribute)
This Pipe is a Gate and is only activated if keys for the field are present.

This class handles nested examples, which nesting level must be indicated using `level`.

## to_tensor_op

[[find in source code]](blob/master/fz_openqa/datamodules/pipelines/collate/field.py#L29)

```python
def to_tensor_op(inputs: List[Any]) -> Tensor:
```
