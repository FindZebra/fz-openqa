# Evaluate Openqa IndentityModel

> Auto-generated documentation for [examples.evaluate_openqa_model](blob/master/examples/evaluate_openqa_model.py) module.

- [Fz-openqa](../README.md#fz-openqa-index) / [Modules](../MODULES.md#fz-openqa-modules) / [Examples](index.md#examples) / Evaluate Openqa IndentityModel
    - [run](#run)

#### Attributes

- `default_cache_dir` - define the default cache location: `Path(fz_openqa.__file__).parent.parent / 'cache'`

## run

[[find in source code]](blob/master/examples/evaluate_openqa_model.py#L39)

```python
@torch.no_grad()
@hydra.main(
    config_path=str(Path(configs.__file__).parent),
    config_name='script_config.yaml',
)
def run(config: DictConfig) -> None:
```
