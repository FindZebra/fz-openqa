# Checkpoint

> Auto-generated documentation for [fz_openqa.inference.checkpoint](blob/master/fz_openqa/inference/checkpoint.py) module.

- [Fz-openqa](../../README.md#fz-openqa-index) / [Modules](../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../index.md#fz-openqa) / [Inference](index.md#inference) / Checkpoint
    - [CheckpointLoader](#checkpointloader)
        - [CheckpointLoader().load_bert](#checkpointloaderload_bert)
        - [CheckpointLoader().load_model](#checkpointloaderload_model)
        - [CheckpointLoader().load_tokenizer](#checkpointloaderload_tokenizer)
        - [CheckpointLoader().model_checkpoint](#checkpointloadermodel_checkpoint)
        - [CheckpointLoader().print_config](#checkpointloaderprint_config)
        - [CheckpointLoader().tokenizer](#checkpointloadertokenizer)
    - [download_asset_from_gdrive](#download_asset_from_gdrive)
    - [get_drive_url](#get_drive_url)
    - [maybe_download_weights](#maybe_download_weights)

## CheckpointLoader

[[find in source code]](blob/master/fz_openqa/inference/checkpoint.py#L44)

```python
class CheckpointLoader():
    def __init__(
        checkpoint_dir: str,
        override=Optional[OmegaConf],
        cache_dir: Optional[str] = None,
    ):
```

### CheckpointLoader().load_bert

[[find in source code]](blob/master/fz_openqa/inference/checkpoint.py#L82)

```python
def load_bert():
```

### CheckpointLoader().load_model

[[find in source code]](blob/master/fz_openqa/inference/checkpoint.py#L85)

```python
def load_model(last=False) -> Model:
```

### CheckpointLoader().load_tokenizer

[[find in source code]](blob/master/fz_openqa/inference/checkpoint.py#L73)

```python
def load_tokenizer():
```

### CheckpointLoader().model_checkpoint

[[find in source code]](blob/master/fz_openqa/inference/checkpoint.py#L60)

```python
def model_checkpoint(last=False) -> Union[None, Path]:
```

### CheckpointLoader().print_config

[[find in source code]](blob/master/fz_openqa/inference/checkpoint.py#L57)

```python
def print_config():
```

### CheckpointLoader().tokenizer

[[find in source code]](blob/master/fz_openqa/inference/checkpoint.py#L76)

```python
@property
def tokenizer() -> PreTrainedTokenizerFast:
```

## download_asset_from_gdrive

[[find in source code]](blob/master/fz_openqa/inference/checkpoint.py#L24)

```python
def download_asset_from_gdrive(
    url: str,
    cache_dir: Optional[str] = None,
    extract: bool = False,
) -> str:
```

## get_drive_url

[[find in source code]](blob/master/fz_openqa/inference/checkpoint.py#L18)

```python
def get_drive_url(url):
```

## maybe_download_weights

[[find in source code]](blob/master/fz_openqa/inference/checkpoint.py#L36)

```python
def maybe_download_weights(
    checkpoint: str,
    cache_dir: Optional[str] = None,
) -> str:
```
