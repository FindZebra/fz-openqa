from abc import ABC
from abc import abstractmethod
from copy import deepcopy
from functools import singledispatchmethod
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import dill

from fz_openqa.datamodules.pipes.control.filter_keys import FilterKey
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.datastruct import Eg
from fz_openqa.utils.fingerprint import get_fingerprint
from fz_openqa.utils.functional import get_batch_eg


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


class Pipe(ABC):
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

    id: Optional[str] = None
    input_filter: Optional[FilterKey] = None
    requires_keys: Optional[List[str]] = None

    def __init__(
        self,
        *,
        id: Optional[str] = None,
        input_filter: Optional[FilterKey] = None,
        update: bool = False,
    ):
        if id is not None:
            self.id = id
        if input_filter is not None:
            self.input_filter = input_filter
        self.update = update

    def output_keys(self, input_keys: List[str]) -> List[str]:
        """
        Return the list of keys that the pipe will output.
        Parameters
        ----------
        input_keys
           The list of keys that the pipe expects as input.

        Returns
        -------
        List[str]
           The list of keys that the pipe will output
        """
        return input_keys

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
        return dill.pickles(self)

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
        return get_fingerprint(x)

    @property
    def fingerprint(self) -> Union[str, Dict[str, Any]]:
        """
        Return a fingerprint(s) of the object.

        Returns
        -------
        Union[str, Dict[str, Any]]
            fingerprint(s) (hex-digested hash of the object),
            allows both a string and a nested structure of strings.
        """
        return self._fingerprint(self)

    def todict(self) -> Dict[str, Any]:
        """
        Return a dictionary representation of the object.

        Returns
        -------
        Dictionary[str, Any]
            Dictionary representation of the object
        """
        data = {"__type__": type(self).__name__, **vars(self)}
        return _filter_null_id(data)

    def __repr__(self) -> str:
        """
        Return a string representation of the object.

        Returns
        -------
        str
            String representation of the object
        """
        return type(self).__name__

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
