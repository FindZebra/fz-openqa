import logging
from abc import ABCMeta
from abc import abstractmethod
from copy import copy
from copy import deepcopy
from functools import singledispatchmethod
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import dill
import rich

from fz_openqa.datamodules.pipes.control.condition import Condition
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.datastruct import Eg
from fz_openqa.utils.fingerprint import get_fingerprint
from fz_openqa.utils.functional import get_batch_eg
from fz_openqa.utils.json_struct import apply_to_json_struct
from fz_openqa.utils.json_struct import reduce_json_struct

logger = logging.getLogger(__name__)


def _filter_null_id(attrs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter out null id.

    Parameters
    ----------
    attrs
        Dictionary of attributes

    Returns
    -------
    Dict[str, Any]
        Dictionary of attributes without null id.
    """
    return {k: v for k, v in attrs.items() if not (k == "id" and v is None)}


class Pipe:
    """
    A pipe is a small unit of computation that ingests,
    modify and returns a batch of data.

    ----------
    Attributes
    id
       An identifier for the pipe.
    input_filter
        Filter used to filter keys in the input data.
    update
        If set to True, output the input batch with the output batch.
    requires_keys
       A list of keys that the pipe requires to be present in the data.
    """

    __metaclass__ = ABCMeta
    id: Optional[str] = None
    input_filter: Optional[Condition] = None
    requires_keys: Optional[List[str]] = None
    _allows_update: bool = True
    _allows_input_filter: bool = True

    def __init__(
        self,
        *,
        id: Optional[str] = None,
        input_filter: Optional[Condition] = None,
        update: bool = False,
    ):
        if not self._allows_update and update:
            raise AttributeError(f"{type(self).__name__} does not allow using update=True")

        if not self._allows_input_filter and input_filter is not None:
            raise AttributeError(f"{type(self).__name__} does not allow using input_filter")

        if id is not None:
            self.id = id
        if input_filter is not None:
            self.input_filter = input_filter
        self.update = update

    def output_keys(self, input_keys: List[str]) -> List[str]:
        """
        Return the list of keys that the pipe is expected to return.
        Parameters
        ----------
        input_keys
           The list of keys that the pipe expects as input.

        Returns
        -------
        List[str]
           The list of keys that the pipe will output
        """
        output_keys = copy(input_keys)
        if self.input_filter is not None:
            output_keys = {k: None for k in output_keys if self.input_filter(k)}.keys()

        if self.update:
            output_keys = {**input_keys, **output_keys}

        return output_keys

    @staticmethod
    def get_eg(batch: Batch, idx: int, filter_op: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Extract example `idx` from a batch, potentially filter the keys

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
        """
        return get_batch_eg(batch=batch, idx=idx, filter_op=filter_op)

    @singledispatchmethod
    def __call__(self, batch: Batch, idx: Optional[List[int]] = None, **kwargs) -> Batch:
        """
        Apply the pipe to a batch of data. Potentially

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
        """

        # filter some input keys
        _batch = self._filter_batch(batch)

        # process the batch
        output = self._call_batch(_batch, idx=idx, **kwargs)

        # update the input batch with the output if update is set to True
        if self.update:
            output = {**batch, **output}

        return output

    def _filter_batch(self, batch: Batch) -> Batch:
        """
        Filter the batch using the input_filter.

        Parameters
        ----------
        batch
            batch to filter

        Returns
        -------
        Batch
            Filtered batch
        """
        if self.input_filter is None:
            return batch

        return {k: v for k, v in batch.items() if self.input_filter(k)}

    @__call__.register(list)
    def _(self, examples: List[Eg], idx: Optional[List[int]] = None, **kwargs) -> Batch:
        """
        Apply the pipe to a list of examples. Typically to concatenate examples.

        Parameters
        ----------
        examples
            batch of examples to apply the pipe to
        idx
            indexes of the examples
        kwargs
            additional arguments

        Returns
        -------
        Batch
            The output batch
        """

        if not all(isinstance(eg, dict) for eg in examples):
            raise TypeError(f"examples must be a list of dicts, got {type(examples[0])}")

        if self.update is True:
            raise AttributeError("Pipe.update is set to True, cannot update a list of examples")

        # filter some input keys
        _egs = list(map(self._filter_batch, examples))

        # process the batch
        return self._call_egs(_egs, idx=idx, **kwargs)

    @abstractmethod
    def _call_batch(self, batch: Batch, idx: Optional[List[int]] = None, **kwargs) -> Batch:
        """
        Main operation applied to the batch.

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
        """
        raise NotImplementedError

    @abstractmethod
    def _call_egs(self, examples: List[Eg], idx: Optional[List[int]] = None, **kwargs) -> Batch:
        """
        Main Operation applied to a list of examples (Egs). Typically to concatenate examples.

        Parameters
        ----------
        examples
            List of examples
        idx
            indexes of the examples
        kwargs
            additional arguments

        Returns
        -------
        Batch
            The output batch
        """
        raise NotImplementedError

    def dill_inspect(self, reduce=True) -> Union[bool, Dict[str, bool]]:
        """
        Inspect the dill representation of the object.

        Parameters
        ----------
        reduce
            Collapse all the dill representations of the object into a single booleans

        Returns
        -------
        Union[bool, Dict[str, Any]
            The dill representation of the object,
            allows both a boolean and a dictionary of booleans.
            If `reduce` is True, the dictionary is collapsed into a single boolean.
        """
        attrs = self._extend_attributes()
        output = {k: dill.pickles(v) for k, v in attrs.items()}
        if reduce:
            output = reduce_json_struct(output, all)
        return output

    @staticmethod
    def _fingerprint(x: Any) -> str:
        """
        Return a fingerprint of the object.

        Parameters
        ----------
        x
            object to fingerprint

        Returns
        -------
        str
            fingerprint (hex-digested hash of the object)
        """
        try:
            return get_fingerprint(x)
        except Exception as ex:
            logger.warning(f"Failed to fingerprint {x}: {ex}")

    @staticmethod
    def safe_fingerprint(x: Any, reduce: bool = False) -> Union[Dict, str]:
        if isinstance(x, Pipe):
            y = x.fingerprint(reduce=reduce)
            rich.print(f"> leaf: {y}")
        else:
            rich.print(f"> not a pipe: {type(x)}")
            return Pipe._fingerprint(x)

    def fingerprint(self, reduce=False) -> Union[str, Dict[str, Any]]:
        """
        Return a fingerprint(s) of the object.

        Returns
        -------
        Union[str, Dict[str, Any]]
            fingerprint(s) (hex-digested hash of the object),
            allows both a string and a nested structure of strings.
        """
        if reduce:
            return self._fingerprint(self)
        else:
            data = self.to_json_struct()
            return apply_to_json_struct(data, get_fingerprint)

    def _extend_attributes(self) -> Dict[str, Any]:
        """
        Returns a dictionary of attributes to extend with self.
        Returns
        -------
        Dict[str, Any]
            Dictionary of attributes to extend with self.
        """
        attrs = {k: v for k, v in vars(self).items()}
        return {"__self__": self, **attrs}

    def to_json_struct(self) -> Dict[str, Any]:
        """
        Return a dictionary representation of the object.

        Returns
        -------
        Dictionary[str, Any]
            Dictionary representation of the object
        """
        data = {"__type__": type(self).__name__, **vars(self)}
        data = _filter_null_id(data)
        data = {k: v.to_json_struct() if isinstance(v, Pipe) else v for k, v in data.items()}
        return data

    def __repr__(self) -> str:
        """
        Return a string representation of the object.

        Returns
        -------
        str
            String representation of the object
        """
        try:
            attrs = [f"{k}={v}" for k, v in vars(self).items()]
        except RecursionError as err:
            import rich

            rich.print(f"#{err} for pipe={type(self)}")
            raise err
        return f"{type(self).__name__}({', '.join(attrs)})"

    def copy(self, **kwargs):
        """
        Return a copy of the object and override the attributes using kwargs.

        Parameters
        ----------
        kwargs
            Attributes to override

        Returns
        -------
        Pipe
            Copy of the object with overridden attributes
        """
        obj = deepcopy(self)
        for k, v in kwargs.items():
            setattr(obj, k, v)
        return obj

    def pprint(self):
        rich.print(self.to_json_struct())
