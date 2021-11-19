# Shape

> Auto-generated documentation for [fz_openqa.utils.shape](blob/master/fz_openqa/utils/shape.py) module.

- [Fz-openqa](../../README.md#fz-openqa-index) / [Modules](../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../index.md#fz-openqa) / [Utils](index.md#utils) / Shape
    - [LeafType](#leaftype)
    - [infer_batch_shape](#infer_batch_shape)
    - [infer_missing_dims](#infer_missing_dims)
    - [infer_shape](#infer_shape)
    - [infer_shape_nested_list](#infer_shape_nested_list)
    - [infer\_shape\_nested\_list\_](#infer_shape_nested_list_)
    - [longest_sublist](#longest_sublist)

## LeafType

[[find in source code]](blob/master/fz_openqa/utils/shape.py#L15)

```python
class LeafType():
    def __init__(leaves: List[Any]):
```

A utility class to represent the type of leaves in a nested list.
Used to returns types in [infer_shape_nested_list_](#infer_shape_nested_list_).

Warning: Hacky implementation, for debugging purposes.

## infer_batch_shape

[[find in source code]](blob/master/fz_openqa/utils/shape.py#L167)

```python
def infer_batch_shape(batch: Batch) -> List[int]:
```

Infer the batch shape, which is the longest common shape of all the fields.
Parameters
----------
batch
    Input batch with nested values.

Returns
-------
List[int]
    Shape of the batch (minimum of all values)

#### See also

- [Batch](datastruct.md#batch)

## infer_missing_dims

[[find in source code]](blob/master/fz_openqa/utils/shape.py#L185)

```python
def infer_missing_dims(n_elements: int, shape: List[int]) -> List[int]:
```

Infer the missing dimensions in a shape.

Parameters
----------
n_elements
    The total number of elements
shape
    Partial shape (e.g. (-1, 8, 6))

Returns
-------
List[int]
    The inferred shape (e.g. (10, 8, 6))

## infer_shape

[[find in source code]](blob/master/fz_openqa/utils/shape.py#L139)

```python
def infer_shape(
    x: Union[List, np.ndarray, Tensor],
    return_leaf_type: bool = False,
) -> Union[Optional[List[int]], Tuple[Optional[List[int]], str]]:
```

Infer shape from nested field.

## infer_shape_nested_list

[[find in source code]](blob/master/fz_openqa/utils/shape.py#L113)

```python
def infer_shape_nested_list(
    x: List[Any],
    return_leaf_type: bool = False,
) -> Union[List[int], Tuple[List[int], LeafType]]:
```

Infer the lengths of nested lists.

Parameters
----------
x
    The nested list to infer the lengths of.
return_leaf_type
    If True, also return the leaf type.

Returns
-------
LUnion[List[int], Tuple[List[int], LeafType]]
    The inferred lengths and optionally the leaf type.

## infer\_shape\_nested\_list\_

[[find in source code]](blob/master/fz_openqa/utils/shape.py#L76)

```python
def infer_shape_nested_list_(
    x,
    level=0,
) -> Tuple[List[int], Optional[LeafType]]:
```

Recursive function used to infer the lengths of nested lists.
See `infer_nested_lengths` for more details.
The leaf level corresponds to the last level of the nested structure.
if sub-fields are not of the same shape, this level is considered to be a leaf level.

## longest_sublist

[[find in source code]](blob/master/fz_openqa/utils/shape.py#L63)

```python
def longest_sublist(lists: List[List[T]]) -> List[T]:
```

Returns the longest common sequence, starting from the start.
