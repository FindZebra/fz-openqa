from numbers import Number
from typing import Any
from typing import List
from typing import Optional
from typing import T
from typing import Tuple
from typing import Union

import numpy as np
from torch import Tensor

from fz_openqa.utils.datastruct import Batch


class LeafType:
    """
    A utility class to represent the type of leaves in a nested list.

    Warning: Hacky implementation, for debugging purposes.
    """

    def __init__(self, leaves: List[Any]):
        self.types: List = []
        self.is_subtype = False
        self.lengths = []

        if all(isinstance(leaf, LeafType) for leaf in leaves):
            self.is_subtype = True

            if all(leaf.is_subtype for leaf in leaves):
                for leaf in leaves:
                    self.types += leaf.types
                    self.lengths += leaf.lengths
            else:
                self.is_subtype = True
                for leaf in leaves:
                    self.types += leaf.types
                    self.lengths += [len(leaf.types)]

            if len(set(self.lengths)) == 1:
                self.is_subtype = False
        else:
            for leaf in leaves:
                self.types.append(type(leaf))

    def __repr__(self):
        if self.is_subtype:
            return (
                f"{type(list())} {min(self.lengths)} to {max(self.lengths)} items, "
                f"value_types={self._repr_types(self.types)}"
            )
        else:
            return self._repr_types(self.types)

    def _repr_types(self, types: List[type]):
        unique_types = set(types)
        return ", ".join(str(t) for t in unique_types)


def longest_sublist(lists: List[List[T]]) -> List[T]:
    """Returns the longest common sequence, starting from the start."""
    output = []
    ref = lists[0]
    for i in range(min(len(le) for le in lists)):
        x = ref[i]
        if all(le[i] == x for le in lists):
            output += [x]
        else:
            break
    return output


def infer_shape_nested_list_(x, level=0) -> Tuple[List[int], Optional[LeafType]]:
    """
    Recursive function used to infer the lengths of nested lists.
    See `infer_nested_lengths` for more details.
    The leaf level corresponds to the last level of the nested structure.
    if sub-fields are not of the same shape, this level is considered to be a leaf level.
    """
    sub_shapes = []
    leaf_types = []
    leaf_level = False
    for y in x:
        if isinstance(y, list):
            sub_shape_y, leaf_type = infer_shape_nested_list_(y, level=level + 1)
            sub_shapes.append(sub_shape_y)
            leaf_types.append(leaf_type)
        else:
            leaf_level = True
            break

    if leaf_level:
        return [len(x)], LeafType(x)
    elif not all(sub_shapes[0] == s for s in sub_shapes):
        # in this case this is a leaf level, but we must transmit shared sub_shapes
        return [len(x), *longest_sublist(sub_shapes)], LeafType(leaf_types)
    else:
        return [len(sub_shapes), *sub_shapes[0]], LeafType(leaf_types)


def infer_shape_nested_list(
    x: List[Any], return_leaf_type: bool = False
) -> Union[List[int], Tuple[List[int], LeafType]]:
    """
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

    """
    shape, leaf_type = infer_shape_nested_list_(x)
    if return_leaf_type:
        return shape, leaf_type
    else:
        return shape


def infer_shape(
    x: Union[List, np.ndarray, Tensor], return_leaf_type: bool = False
) -> Union[Optional[List[int]], Tuple[Optional[List[int]], str]]:
    """Infer shape from nested field."""
    if isinstance(x, Tensor):
        shape = list(x.shape)
        if return_leaf_type:
            return shape, f"{type(x)} (dtype={x.dtype}, device={x.device})"
        else:
            return shape
    elif isinstance(x, np.ndarray):
        shape = list(x.shape)
        if return_leaf_type:
            return shape, f"{type(x)} (dtype=np.{x.dtype})"
        else:
            return shape
    elif isinstance(x, (Number, str)) or x is None:
        if return_leaf_type:
            return [], str(type(x))
        else:
            return []
    elif isinstance(x, list):
        return infer_shape_nested_list(x, return_leaf_type=return_leaf_type)
    else:
        raise TypeError(f"Unsupported type {type(x)}")


def infer_batch_shape(batch: Batch) -> List[int]:
    """
    Infer the batch shape, which is the longest common shape of all the fields.
    Parameters
    ----------
    batch
        Input batch with nested values.

    Returns
    -------
    List[int]
        Shape of the batch (minimum of all values)
    """
    shapes = [infer_shape(b) for b in batch.values()]
    shapes = [s for s in shapes if s is not None and len(s) > 0]
    return longest_sublist(shapes)


def infer_missing_dims(n_elements: int, *, shape: List[int]) -> List[int]:
    """
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
    """
    assert all(y != 0 for y in shape)
    if all(y > 0 for y in shape):
        return shape
    else:
        neg_idx = [i for i, y in enumerate(shape) if y < 0]
        assert len(neg_idx) == 1, "Only one dimension can be negative"
        neg_idx = neg_idx[0]
        known_dims = [y for y in shape if y > 0]

        p = np.prod(known_dims)
        assert n_elements % p == 0, "n_elements must be divisible by product of known dimensions"
        missing_dim = n_elements // p
        shape[neg_idx] = missing_dim

    return shape
