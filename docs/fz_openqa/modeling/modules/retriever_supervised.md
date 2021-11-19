# RetrieverSupervised

> Auto-generated documentation for [fz_openqa.modeling.modules.retriever_supervised](blob/master/fz_openqa/modeling/modules/retriever_supervised.py) module.

- [Fz-openqa](../../../README.md#fz-openqa-index) / [Modules](../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../index.md#fz-openqa) / [Modeling](../index.md#modeling) / [Modules](index.md#modules) / RetrieverSupervised
    - [RetrieverSupervised](#retrieversupervised)

## RetrieverSupervised

[[find in source code]](blob/master/fz_openqa/modeling/modules/retriever_supervised.py#L21)

```python
class RetrieverSupervised(Module):
    def __init__(
        similarity: Union[DictConfig, Similarity] = DotProduct(),
        **kwargs,
    ):
```

#### Attributes

- `pbar_metrics` - metrics to display in the progress bar: `['train/retriever/Accuracy', 'validation/retrie...`

#### See also

- [Module](base.md#module)
