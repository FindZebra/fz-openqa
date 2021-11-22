# Heads

> Auto-generated documentation for [fz_openqa.modeling.layers.heads](blob/master/fz_openqa/modeling/layers/heads.py) module.

- [Fz-openqa](../../../README.md#fz-openqa-index) / [Modules](../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../index.md#fz-openqa) / [Modeling](../index.md#modeling) / [Layers](index.md#layers) / Heads
    - [cls_head](#cls_head)
        - [cls_head().forward](#cls_headforward)

## cls_head

[[find in source code]](blob/master/fz_openqa/modeling/layers/heads.py#L7)

```python
class cls_head(nn.Module):
    def __init__(
        bert: BertPreTrainedModel,
        output_size: int,
        normalize: bool = True,
    ):
```

A linear head consuming the representation at the CLS token

### cls_head().forward

[[find in source code]](blob/master/fz_openqa/modeling/layers/heads.py#L20)

```python
def forward(last_hidden_state: Tensor):
```
