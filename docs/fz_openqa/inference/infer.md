# Infer

> Auto-generated documentation for [fz_openqa.inference.infer](blob/master/fz_openqa/inference/infer.py) module.

- [Fz-openqa](../../README.md#fz-openqa-index) / [Modules](../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../index.md#fz-openqa) / [Inference](index.md#inference) / Infer
    - [encode_data](#encode_data)
    - [load_and_infer](#load_and_infer)
    - [load_inputs](#load_inputs)
    - [load_model_from_checkpoint](#load_model_from_checkpoint)

## encode_data

[[find in source code]](blob/master/fz_openqa/inference/infer.py#L95)

```python
def encode_data(data, tokenizer):
```

Encode the data retrieve form a .json file

## load_and_infer

[[find in source code]](blob/master/fz_openqa/inference/infer.py#L26)

```python
@hydra.main(config_path='../configs/', config_name='infer_config.yaml')
def load_and_infer(config: DictConfig) -> Dict[str, float]:
```

Load a train Module and process some input data.
The input data is a source .json file if provided else this is
the test set of the datamodule described in the config file.

NB: implemented for a reader Module only
todo: retriever evaluation
todo: full Module evaluation
todo: loader both a reader and a retriever

## load_inputs

[[find in source code]](blob/master/fz_openqa/inference/infer.py#L114)

```python
def load_inputs(path: str) -> Dict[str, List[Any]]:
```

Load data from a .json file

## load_model_from_checkpoint

[[find in source code]](blob/master/fz_openqa/inference/infer.py#L124)

```python
def load_model_from_checkpoint(
    cls: pl.LightningModule.__class__,
    path: str,
    device: torch.device,
    **kwargs,
):
```

load the Module form a checkpoint
