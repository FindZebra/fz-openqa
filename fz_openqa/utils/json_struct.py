"""
Functions to process json-like structures
"""
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import T
from typing import Union


def apply_to_json_struct(data: Union[List, Dict], fn: Callable, **kwargs) -> Union[List, Dict]:
    """
    Apply a function to a json-like structure
    Parameters
    ----------
    data
        json-like structure
    fn
        function to apply
    kwargs
        keyword arguments to pass to fn

    Returns
    -------
    json-like structure
    """
    if isinstance(data, dict):
        try:
            output = {
                key: apply_to_json_struct(value, fn, key=key, **kwargs)
                for key, value in data.items()
            }
        except Exception:
            output = {key: apply_to_json_struct(value, fn, **kwargs) for key, value in data.items()}

        return output
    elif isinstance(data, list):
        return [apply_to_json_struct(value, fn, **kwargs) for value in data]
    else:
        return fn(data, **kwargs)


def flatten_json_struct(data: Union[List, Dict]) -> Iterable[Any]:
    """
    Flatten a json-like structure
    Parameters
    ----------
    data
        json-like structure
    Yields
    -------
    Any
        Leaves of json-like structure
    """
    if isinstance(data, dict):
        for x in data.values():
            for leaf in flatten_json_struct(x):
                yield leaf
    elif isinstance(data, list):
        for x in data:
            for leaf in flatten_json_struct(x):
                yield leaf
    else:
        yield data


def reduce_json_struct(data: Union[List, Dict], reduce_op: Callable[[Iterable[T]], T]) -> T:
    """
    Reduce a json-like structure
    Parameters
    ----------
    data
        json-like structure
    reduce_op
        reduce operation
    Returns
    -------
    reduced json-like structure
    """
    leaves = flatten_json_struct(data)
    return reduce_op(leaves)
