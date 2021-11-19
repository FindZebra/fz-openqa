# General

> Auto-generated documentation for [fz_openqa.datamodules.pipelines.collate.general](blob/master/fz_openqa/datamodules/pipelines/collate/general.py) module.

- [Fz-openqa](../../../../README.md#fz-openqa-index) / [Modules](../../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../../index.md#fz-openqa) / [Datamodules](../../index.md#datamodules) / [Pipelines](../index.md#pipelines) / [Collate](index.md#collate) / General
    - [CollateAsTensor](#collateastensor)
    - [CollateTokens](#collatetokens)

## CollateAsTensor

[[find in source code]](blob/master/fz_openqa/datamodules/pipelines/collate/general.py#L18)

```python
class CollateAsTensor(Sequential):
    def __init__(keys: List[str], id: str = 'collate-simple-attrs'):
```

A pipeline to concatenate simple attributes and cast as tensors

## CollateTokens

[[find in source code]](blob/master/fz_openqa/datamodules/pipelines/collate/general.py#L29)

```python
class CollateTokens(Sequential):
    def __init__(
        prefix: str,
        tokenizer: PreTrainedTokenizerFast,
        shape: Optional[List[int]] = None,
        id: Optional[str] = None,
    ):
```

A pipeline to collate token fields (*.input_ids, *.attention_mask).
