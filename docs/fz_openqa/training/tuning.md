# Tuning

> Auto-generated documentation for [fz_openqa.training.tuning](blob/master/fz_openqa/training/tuning.py) module.

- [Fz-openqa](../../README.md#fz-openqa-index) / [Modules](../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../index.md#fz-openqa) / [Training](index.md#training) / Tuning
    - [format_key](#format_key)
    - [run_tune](#run_tune)
    - [run_tune_with_config](#run_tune_with_config)
    - [trial](#trial)

## format_key

[[find in source code]](blob/master/fz_openqa/training/tuning.py#L27)

```python
def format_key(k, globals=DEFAULT_GLOBALS):
```

#### See also

- [DEFAULT_GLOBALS](#default_globals)

## run_tune

[[find in source code]](blob/master/fz_openqa/training/tuning.py#L89)

```python
@hydra.main(config_path='../configs/', config_name='hpo_config.yaml')
def run_tune(config: DictConfig) -> None:
```

## run_tune_with_config

[[find in source code]](blob/master/fz_openqa/training/tuning.py#L49)

```python
def run_tune_with_config(config: DictConfig):
```

## trial

[[find in source code]](blob/master/fz_openqa/training/tuning.py#L34)

```python
def trial(args, checkpoint_dir=None, **kwargs):
```
