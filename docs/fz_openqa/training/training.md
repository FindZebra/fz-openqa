# Training

> Auto-generated documentation for [fz_openqa.training.training](blob/master/fz_openqa/training/training.py) module.

- [Fz-openqa](../../README.md#fz-openqa-index) / [Modules](../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../index.md#fz-openqa) / [Training](index.md#training) / Training
    - [train](#train)

## train

[[find in source code]](blob/master/fz_openqa/training/training.py#L21)

```python
def train(config: DictConfig) -> Optional[float]:
```

Contains training pipeline.
Instantiates all PyTorch Lightning objects from config.

#### Arguments

- `config` *DictConfig* - Configuration composed by Hydra.

#### Returns

- `Optional[float]` - Metric score for hyperparameter optimization.
