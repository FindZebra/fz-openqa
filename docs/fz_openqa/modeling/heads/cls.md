# Cls

> Auto-generated documentation for [fz_openqa.modeling.heads.cls](blob/master/fz_openqa/modeling/heads/cls.py) module.

- [Fz-openqa](../../../README.md#fz-openqa-index) / [Modules](../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../index.md#fz-openqa) / [Modeling](../index.md#modeling) / [Heads](index.md#heads) / Cls
    - [ClsHead](#clshead)
        - [ClsHead().forward](#clsheadforward)

## ClsHead

[[find in source code]](blob/master/fz_openqa/modeling/heads/cls.py#L11)

```python
class ClsHead(Head):
    def __init__(
        bert: BertPreTrainedModel,
        output_size: Optional[int],
        normalize: bool = False,
    ):
```

#### See also

- [Head](base.md#head)

### ClsHead().forward

[[find in source code]](blob/master/fz_openqa/modeling/heads/cls.py#L23)

```python
def forward(last_hidden_state: Tensor, **kwargs) -> Tensor:
```
