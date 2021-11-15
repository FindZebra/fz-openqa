import logging
from abc import ABCMeta
from copy import deepcopy
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import dill
import rich

from fz_openqa.utils.fingerprint import get_fingerprint
from fz_openqa.utils.json_struct import apply_to_json_struct
from fz_openqa.utils.json_struct import reduce_json_struct

logger = logging.getLogger(__name__)


def leaf_to_json_struct(v: Any) -> Union[Dict, List]:
    """Convert a leaf value into a json structure."""
    if isinstance(v, Component):
        return v.to_json_struct()
    elif isinstance(v, list):
        return [leaf_to_json_struct(x) for x in v]
    elif isinstance(v, dict):
        return {k: leaf_to_json_struct(x) for k, x in v.items()}
    else:
        return v


class Component:
    """
    A Component is an object used to process data.
    It supports a few methods so it can nicely be integrated into the data
    processing pipeline (multiprocessing, caching).
    A component must be safely and deterministically serializable using pickle.

    Functionalities:
     - A component serialization can be inspected using `dill_inspect()`
     - The hash of the object can be obtained using `fingerprint()`
     - The object can be reduced to a json-struct using `to_json_struct()`
     - The object can be copied using `copy()`
     - The object can be printed using `pprint()`


    ----------
    Attributes
    id
       An identifier for the component.
    """

    __metaclass__ = ABCMeta
    id: Optional[str] = None

    def __init__(
        self,
        *,
        id: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        id
           An identifier for the pipe.
        """
        if id is not None:
            self.id = id

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
        attrs = {"__self__": self, **self._get_attributes()}
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
        if isinstance(x, Component):
            return x.fingerprint(reduce=reduce)
        else:
            return Component._fingerprint(x)

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

            def maybe_get_fingerprint(v: Any, key: str) -> str:
                """return the fingerprint, excepts if key==__name__"""
                if key == "__name__":
                    return v
                else:
                    return get_fingerprint(v)

            return apply_to_json_struct(data, maybe_get_fingerprint)

    def to_json_struct(self) -> Dict[str, Any]:
        """
        Return a dictionary representation of the object.

        Returns
        -------
        Dictionary[str, Any]
            Dictionary representation of the object
        """
        attributes = self._get_attributes()

        data = {"__name__": type(self).__name__, **attributes}
        data = {k: v for k, v in data.items() if not (k == "id" and v is None)}
        data = {k: leaf_to_json_struct(v) for k, v in data.items()}
        return data

    def _get_attributes(self) -> Dict:
        """
        Return a dictionary of attributes of the object, uses __getstate__ if available.

        Returns
        -------
        Dict
            Dictionary of attributes of the object
        """
        if hasattr(self, "__getstate__"):
            attributes = self.__getstate__()
        else:
            attributes = vars(self)
        return attributes

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
        Component
            Copy of the object with overridden attributes
        """
        obj = deepcopy(self)
        for k, v in kwargs.items():
            setattr(obj, k, v)
        return obj

    def pprint(self):
        rich.print(self.to_json_struct())
