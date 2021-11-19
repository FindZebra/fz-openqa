# Base

> Auto-generated documentation for [fz_openqa.datamodules.pipes.base](blob/master/fz_openqa/datamodules/pipes/base.py) module.

- [Fz-openqa](../../../README.md#fz-openqa-index) / [Modules](../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../index.md#fz-openqa) / [Datamodules](../index.md#datamodules) / [Pipes](index.md#pipes) / Base
    - [Pipe](#pipe)
        - [Pipe().\_\_call\_\_](#pipe__call__)
        - [Pipe.get_eg](#pipeget_eg)
        - [Pipe().output_keys](#pipeoutput_keys)

## Pipe

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/base.py#L21)

```python
class Pipe(Component):
    def __init__(
        id: Optional[str] = None,
        input_filter: Optional[Condition] = None,
        update: bool = False,
    ):
```

A pipe is a small unit of computation that ingests,
modify and returns a batch of data.

----------
Attributes
id
   An identifier for the pipe.
input_filter
    Condition used to filter keys in the input data.
update
    If set to True, output the input batch with the output batch.
requires_keys
   A list of keys that the pipe requires to be present in the data.

#### See also

- [Component](../component.md#component)

### Pipe().\_\_call\_\_

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/base.py#L117)

```python
@singledispatchmethod
def __call__(
    batch: Batch,
    idx: Optional[List[int]] = None,
    **kwargs,
) -> Batch:
```

Apply the pipe to a batch of data. Potentially filter the keys using the input_filter.
The output of `_call_batch()` is used to update the input batch (before filtering)
if update=True, else the raw output is returned.

Parameters
----------
batch
    batch to apply the pipe to
idx
    indexes of the batch examples
kwargs
    additional arguments

Returns
-------
Batch
    The output batch

#### See also

- [Batch](../../utils/datastruct.md#batch)

### Pipe.get_eg

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/base.py#L96)

```python
@staticmethod
def get_eg(
    batch: Batch,
    idx: int,
    filter_op: Optional[Callable] = None,
) -> Dict[str, Any]:
```

Extract example `idx` from a batch, potentially filter keys.

Parameters
----------
batch
   Input batch
idx
   Index of the example to extract
filter_op
   A function that used to filter the keys

Returns
-------
Dict[str, Any]
   The example of rank `idx`

#### See also

- [Batch](../../utils/datastruct.md#batch)

### Pipe().output_keys

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/base.py#L74)

```python
def output_keys(input_keys: List[str]) -> List[str]:
```

Return the list of keys that the pipe is expected to return.
Parameters
----------
input_keys
   The list of keys that the pipe expects as input.

Returns
-------
List[str]
   The list of keys that the pipe will output
