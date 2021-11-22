# Base

> Auto-generated documentation for [fz_openqa.modeling.heads.base](blob/master/fz_openqa/modeling/heads/base.py) module.

- [Fz-openqa](../../../README.md#fz-openqa-index) / [Modules](../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../index.md#fz-openqa) / [Modeling](../index.md#modeling) / [Heads](index.md#heads) / Base
    - [Head](#head)
        - [Head().forward](#headforward)

## Head

[[find in source code]](blob/master/fz_openqa/modeling/heads/base.py#L9)

```python
class Head(nn.Module, ABC):
    def __init__(bert: BertPreTrainedModel, output_size: int):
```

### Head().forward

[[find in source code]](blob/master/fz_openqa/modeling/heads/base.py#L15)

```python
@abstractmethod
def forward(last_hidden_state: Tensor, **kwargs) -> Tensor:
```
