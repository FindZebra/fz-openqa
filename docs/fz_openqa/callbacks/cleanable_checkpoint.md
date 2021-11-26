# CleanableCheckpoint

> Auto-generated documentation for [fz_openqa.callbacks.cleanable_checkpoint](blob/master/fz_openqa/callbacks/cleanable_checkpoint.py) module.

- [Fz-openqa](../../README.md#fz-openqa-index) / [Modules](../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../index.md#fz-openqa) / [Callbacks](index.md#callbacks) / CleanableCheckpoint
    - [CleanableCheckpoint](#cleanablecheckpoint)
        - [CleanableCheckpoint().on_test_epoch_end](#cleanablecheckpointon_test_epoch_end)

## CleanableCheckpoint

[[find in source code]](blob/master/fz_openqa/callbacks/cleanable_checkpoint.py#L7)

```python
class CleanableCheckpoint(ModelCheckpoint):
    def __init__(cleanup_threshold: str = None, *args, **kwargs):
```

### CleanableCheckpoint().on_test_epoch_end

[[find in source code]](blob/master/fz_openqa/callbacks/cleanable_checkpoint.py#L12)

```python
def on_test_epoch_end(
    trainer: 'pl.Trainer',
    pl_module: 'pl.LightningModule',
) -> None:
```
