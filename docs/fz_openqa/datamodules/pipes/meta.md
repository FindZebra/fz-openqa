# Meta

> Auto-generated documentation for [fz_openqa.datamodules.pipes.meta](blob/master/fz_openqa/datamodules/pipes/meta.py) module.

- [Fz-openqa](../../../README.md#fz-openqa-index) / [Modules](../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../index.md#fz-openqa) / [Datamodules](../index.md#datamodules) / [Pipes](index.md#pipes) / Meta
    - [BlockSequential](#blocksequential)
        - [BlockSequential().output_keys](#blocksequentialoutput_keys)
    - [Gate](#gate)
        - [Gate().is_switched_on](#gateis_switched_on)
        - [Gate().output_keys](#gateoutput_keys)
    - [MetaPipe](#metapipe)
    - [Parallel](#parallel)
        - [Parallel().output_keys](#paralleloutput_keys)
    - [ParallelbyField](#parallelbyfield)
    - [PipeProcessError](#pipeprocesserror)
    - [Sequential](#sequential)
        - [Sequential().output_keys](#sequentialoutput_keys)

## BlockSequential

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/meta.py#L191)

```python
class BlockSequential(MetaPipe):
    def __init__(blocks: List[Tuple[str, Pipe]], **kwargs):
```

A sequence of Pipes organized into blocks

#### See also

- [MetaPipe](#metapipe)

### BlockSequential().output_keys

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/meta.py#L206)

```python
def output_keys(input_keys: List[str]) -> List[str]:
```

## Gate

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/meta.py#L130)

```python
class Gate(MetaPipe):
    def __init__(
        condition: Union[bool, Callable],
        pipe: Optional[Pipe],
        alt: Optional[Pipe] = None,
        **kwargs,
    ):
```

Execute the pipe if the condition is valid, else execute alt.

#### See also

- [MetaPipe](#metapipe)

### Gate().is_switched_on

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/meta.py#L183)

```python
def is_switched_on(batch):
```

### Gate().output_keys

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/meta.py#L153)

```python
def output_keys(input_keys: List[str]) -> List[str]:
```

## MetaPipe

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/meta.py#L53)

```python
class MetaPipe(Pipe):
```

A class that executes other pipes (Sequential, Parallel)

## Parallel

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/meta.py#L94)

```python
class Parallel(Sequential):
```

Execute pipes in parallel and merge the outputs

#### See also

- [Sequential](#sequential)

### Parallel().output_keys

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/meta.py#L121)

```python
def output_keys(input_keys: List[str]) -> List[str]:
```

## ParallelbyField

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/meta.py#L212)

```python
class ParallelbyField(Parallel):
    def __init__(pipes: Dict[str, Pipe], **kwargs):
```

Run a pipe for each field

#### See also

- [Parallel](#parallel)

## PipeProcessError

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/meta.py#L18)

```python
class PipeProcessError(Exception):
    def __init__(meta_pipe: Pipe, pipe: Pipe, batch: Batch, **kwargs):
```

Base class for other exceptions

#### See also

- [Batch](../../utils/datastruct.md#batch)

## Sequential

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/meta.py#L74)

```python
class Sequential(MetaPipe):
    def __init__(*pipes: Optional[Union[Callable, Pipe]], **kwargs):
```

Execute a sequence of pipes.

#### See also

- [MetaPipe](#metapipe)

### Sequential().output_keys

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/meta.py#L88)

```python
def output_keys(input_keys: List[str]) -> List[str]:
```
