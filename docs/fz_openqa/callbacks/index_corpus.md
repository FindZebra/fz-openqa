# IndexCorpus

> Auto-generated documentation for [fz_openqa.callbacks.index_corpus](blob/master/fz_openqa/callbacks/index_corpus.py) module.

- [Fz-openqa](../../README.md#fz-openqa-index) / [Modules](../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../index.md#fz-openqa) / [Callbacks](index.md#callbacks) / IndexCorpus
    - [AcceleratorWrapper](#acceleratorwrapper)
    - [IndexCorpus](#indexcorpus)
        - [IndexCorpus().on_validation_start](#indexcorpuson_validation_start)

## AcceleratorWrapper

[[find in source code]](blob/master/fz_openqa/callbacks/index_corpus.py#L10)

```python
class AcceleratorWrapper():
    def __init__(trainer: pl.Trainer):
```

## IndexCorpus

[[find in source code]](blob/master/fz_openqa/callbacks/index_corpus.py#L19)

```python
class IndexCorpus(Callback):
```

### IndexCorpus().on_validation_start

[[find in source code]](blob/master/fz_openqa/callbacks/index_corpus.py#L23)

```python
@torch.no_grad()
def on_validation_start(
    trainer: 'pl.Trainer',
    pl_module: 'pl.LightningModule',
) -> None:
```

Compute the corpus vectors using the model.
