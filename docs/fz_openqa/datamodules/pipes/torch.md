# Torch

> Auto-generated documentation for [fz_openqa.datamodules.pipes.torch](blob/master/fz_openqa/datamodules/pipes/torch.py) module.

- [Fz-openqa](../../../README.md#fz-openqa-index) / [Modules](../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../index.md#fz-openqa) / [Datamodules](../index.md#datamodules) / [Pipes](index.md#pipes) / Torch
    - [Forward](#forward)
    - [Itemize](#itemize)
        - [Itemize().itemize](#itemizeitemize)
    - [ToNumpy](#tonumpy)

## Forward

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/torch.py#L40)

```python
class Forward(Pipe):
    def __init__(model: Union[Callable, torch.nn.Module], **kwargs):
```

Process a batch of data using a model: output[key] = model(batch)

## Itemize

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/torch.py#L25)

```python
class Itemize(Pipe):
```

Convert all values to lists.

### Itemize().itemize

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/torch.py#L28)

```python
def itemize(values: Any):
```

## ToNumpy

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/torch.py#L14)

```python
class ToNumpy(Pipe):
    def __init__(as_contiguous: bool = True, **kwargs):
```

Move Tensors to the CPU and cast to numpy arrays.
