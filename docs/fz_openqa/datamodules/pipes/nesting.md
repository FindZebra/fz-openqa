# Nesting

> Auto-generated documentation for [fz_openqa.datamodules.pipes.nesting](blob/master/fz_openqa/datamodules/pipes/nesting.py) module.

- [Fz-openqa](../../../README.md#fz-openqa-index) / [Modules](../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../index.md#fz-openqa) / [Datamodules](../index.md#datamodules) / [Pipes](index.md#pipes) / Nesting
    - [ApplyAsFlatten](#applyasflatten)
    - [Flatten](#flatten)
    - [Nest](#nest)
        - [Nest.nest](#nestnest)
    - [Nested](#nested)
    - [NestedLevel1](#nestedlevel1)

## ApplyAsFlatten

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/nesting.py#L103)

```python
class ApplyAsFlatten(Pipe):
    def __init__(pipe: Pipe, level: int = 1, **kwargs):
```

Flattens the first `level+1` batch dimensions and
applies the pipe to the flattened batch.

Warning: Do not use this pipe if the inner pipe drops nested values
or modifies the order of the batch elements!

Notes
-------
This pipe is equivalent to:

```python
# example data
h = (20, 10) # some vector dimension
nested_shape = (10, 8, 8) # some nested batch dimension
batch = np.random.randn(size=([nested_shape, *h)]

# ApplyAsFlatten(pipe)
batch = batch.reshape(-1, *h)
batch = pipe(batch)
batch = batch.reshape(*nested_shape, *h)
```

## Flatten

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/nesting.py#L24)

```python
class Flatten(ApplyToAll):
    def __init__(level: int = 1, **kwargs):
```

Flatten a nested batch up to dimension=`level`.
For instance a batch of shape (x, 3, 4, ...) with level=2 will be flattened to (x *3 * 4, ...)

#### See also

- [ApplyToAll](basic.md#applytoall)

## Nest

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/nesting.py#L46)

```python
class Nest(ApplyToAll):
    def __init__(shape: Optional[List[int]], **kwargs):
```

Nest a flat batch. This is equivalent to calling np.reshape to all values,
except that this method can handle np.ndarray, Tensors and lists.
If the target shape is unknown at initialization time, the `shape` attributed
can be passed as a keyword argument to the __call__ method.

#### See also

- [ApplyToAll](basic.md#applytoall)

### Nest.nest

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/nesting.py#L66)

```python
@staticmethod
def nest(
    x: T,
    _shape: Optional[List[int]],
    shape: Optional[List[int]] = None,
    **kwargs,
) -> T:
```

Nest the input x according to shape or _shape.
This allows specifying a shape that is not known at init.

Parameters
----------
x
    Input to nest.
shape
    Primary and optional target shape of the nested batch
_shape
    Secondary and optional target shape of the nested batch

Returns
-------
Union[List, Tensor, np.ndarray]
    Nested input.

## Nested

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/nesting.py#L201)

```python
class Nested(ApplyAsFlatten):
    def __init__(pipe: Pipe, level=1, **kwargs):
```

Apply a pipe to each nested value up to dimension `level`.
This can be use to modify the nested field inplace  (i.e. sorting, deleting).
However the all pipe output must have the same batch size.

#### See also

- [ApplyAsFlatten](#applyasflatten)

## NestedLevel1

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/nesting.py#L166)

```python
class NestedLevel1(Pipe):
    def __init__(pipe: Pipe, **kwargs):
```

Apply a pipe to each nested value, handling each nested field as a separate batch.
This can be use to modify the nested field inplace  (i.e. sorting, deleting).
However the all pipe output must have the same batch size.
