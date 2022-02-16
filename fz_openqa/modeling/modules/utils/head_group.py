import json
from copy import deepcopy
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional

from torch import nn

from fz_openqa.modeling.heads.base import Head


class HeadGroup(nn.ModuleDict):
    """A class representing a dictionary of heads.
    Heads parameters can be shared via the argument `mapping`."""

    def __init__(self, head: Head, *, keys: List[str], mapping: Optional[Dict[str, str]] = None):
        if mapping is None:
            mapping = {k: k for k in keys}
        super().__init__({k: deepcopy(head) for k in set(mapping.values())})
        self._head_mapping = mapping

    def __getitem__(self, key) -> Head:
        key = self._head_mapping[key]
        return super().__getitem__(key)  # type: ignore

    def values(self) -> Iterable[Head]:
        return (self.__getitem__(k) for k in self.keys())

    def keys(self) -> Iterable[str]:
        return self._head_mapping.keys()

    def __len__(self) -> int:
        return len(list(self.keys()))

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._modules

    def __repr__(self):
        u = super().__repr__()
        u = u[:-1]
        u += f"\n  mapping={json.dumps(self._head_mapping, indent=4)}\n)"
        return u
