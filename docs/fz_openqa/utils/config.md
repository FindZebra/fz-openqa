# Config

> Auto-generated documentation for [fz_openqa.utils.config](blob/master/fz_openqa/utils/config.py) module.

- [Fz-openqa](../../README.md#fz-openqa-index) / [Modules](../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../index.md#fz-openqa) / [Utils](index.md#utils) / Config
    - [print_config](#print_config)
    - [resolve_config_paths](#resolve_config_paths)

## print_config

[[find in source code]](blob/master/fz_openqa/utils/config.py#L27)

```python
@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Optional[Sequence[str]] = None,
    resolve: bool = True,
) -> None:
```

Prints content of DictConfig using Rich library and its tree structure.

#### Arguments

- `config` *DictConfig* - Configuration composed by Hydra.
- `fields` *Sequence[str], optional* - Determines which main fields from config will
be printed and in what order.
- `resolve` *bool, optional* - Whether to resolve reference fields of DictConfig.

## resolve_config_paths

[[find in source code]](blob/master/fz_openqa/utils/config.py#L17)

```python
def resolve_config_paths(
    config: DictConfig,
    path: str = '',
    excludes: List[str] = ['hydra'],
):
```
