# Train Utils

> Auto-generated documentation for [fz_openqa.utils.train_utils](blob/master/fz_openqa/utils/train_utils.py) module.

- [Fz-openqa](../../README.md#fz-openqa-index) / [Modules](../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../index.md#fz-openqa) / [Utils](index.md#utils) / Train Utils
    - [empty](#empty)
    - [extras](#extras)
    - [finish](#finish)
    - [get_logger](#get_logger)
    - [log_hyperparameters](#log_hyperparameters)
    - [setup_safe_env](#setup_safe_env)

## empty

[[find in source code]](blob/master/fz_openqa/utils/train_utils.py#L19)

```python
def empty(*args, **kwargs):
```

## extras

[[find in source code]](blob/master/fz_openqa/utils/train_utils.py#L124)

```python
def extras(config: DictConfig) -> None:
```

A couple of optional utilities, controlled by main config file:
- disabling warnings
- easier access to debug mode
- forcing debug friendly configuration
- forcing multi-gpu friendly configuration
Modifies DictConfig in place.

#### Arguments

- `config` *DictConfig* - Configuration composed by Hydra.

## finish

[[find in source code]](blob/master/fz_openqa/utils/train_utils.py#L45)

```python
def finish(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
```

Makes sure everything closed properly.

## get_logger

[[find in source code]](blob/master/fz_openqa/utils/train_utils.py#L23)

```python
def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
```

Initializes multi-GPU-friendly python logger.

## log_hyperparameters

[[find in source code]](blob/master/fz_openqa/utils/train_utils.py#L61)

```python
@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
```

This method controls which parameters from Hydra config are saved by Lightning loggers.

Additionaly saves:
    - number of trainable model parameters

## setup_safe_env

[[find in source code]](blob/master/fz_openqa/utils/train_utils.py#L155)

```python
def setup_safe_env():
```
