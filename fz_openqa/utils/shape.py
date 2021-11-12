from numbers import Number
from operator import itemgetter
from typing import Any
from typing import List
from typing import Optional
from typing import T
from typing import Tuple
from typing import Union

import numpy as np
import torch
from torch import Tensor

from fz_openqa.utils.datastruct import Batch


class LeafType:
    """
    A class to represent the type of leaves in a nested list.

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

                if len(set(self.lengths)) == 1:
                    self.is_subtype = False
            else:
                self.is_subtype = True
                for leaf in leaves:
                    self.types += leaf.types
                    self.lengths += [len(leaf.types)]
        else:
            for leaf in leaves:
                self.types.append(type(leaf))

    def __repr__(self):
        if self.is_subtype:
            return (
                f"{type(list())}, {min(self.lengths)} to {max(self.lengths)} items, "
                f"value_types={self._repr_types(self.types)}"
            )
        else:
            return self._repr_types(self.types)

    def _repr_types(self, types: List[type]):
        unique_types = set(types)
        if len(unique_types) == 1:
            return str(types[0])
        else:
            type_strs = ", ".join(str(t) for t in unique_types)
            return f"MixedTypes({type_strs})"


def longest_sublist(lists: List[List[T]]) -> List[T]:
    """Returns the longest common sequence, starting from the start."""
    output = []
    for i in range(min(len(le) for le in lists)):
        x = lists[0][i]
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
            return shape, f"<dtype={x.dtype}, device={x.device}>"
        else:
            return shape
    elif isinstance(x, np.ndarray):
        shape = list(x.shape)
        if return_leaf_type:
            return shape, f"<dtype=np.{x.dtype}>"
        else:
            return shape
    elif isinstance(x, (Number, str)) or x is None:
        if return_leaf_type:
            return None, str(type(x))
        else:
            return None
    elif isinstance(x, list):
        return infer_shape_nested_list(x, return_leaf_type=return_leaf_type)
    else:
        raise TypeError(f"Unsupported type {type(x)}")


def infer_min_shape(batch: Batch) -> List[int]:
    """
    Infer minimum shape from nested fields in a batch.
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
    shapes = filter(None, shapes)
    min_dim, shape = min([(len(s), s) for s in shapes], key=itemgetter(0))
    return shape
