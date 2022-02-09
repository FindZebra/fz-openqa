# Sample Lm

> Auto-generated documentation for [fz_openqa.callbacks.sample_lm](blob/master/fz_openqa/callbacks/sample_lm.py) module.

- [Fz-openqa](../../README.md#fz-openqa-index) / [Modules](../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../index.md#fz-openqa) / [Callbacks](index.md#callbacks) / Sample Lm
    - [LogPredictions](#samplelanguagemodel)
        - [LogPredictions().display](#samplelanguagemodeldisplay)
        - [LogPredictions().on_epoch_start](#samplelanguagemodelon_epoch_start)

## LogPredictions

[[find in source code]](blob/master/fz_openqa/callbacks/sample_lm.py#L9)

```python
class LogPredictions(Callback):
```

### LogPredictions().display

[[find in source code]](blob/master/fz_openqa/callbacks/sample_lm.py#L16)

```python
def display(input_ids, tokenizer):
```

### LogPredictions().on_epoch_start

[[find in source code]](blob/master/fz_openqa/callbacks/sample_lm.py#L10)

```python
def on_epoch_start(
    trainer: 'pl.Trainer',
    pl_module: 'pl.LightningModule',
) -> None:
```
