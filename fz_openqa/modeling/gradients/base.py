import abc
from enum import Enum
from typing import Dict

from torch import nn


class Gradients(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    @abc.abstractmethod
    def __call__(self, **kwargs) -> Dict:
        ...

    def step(self, **data) -> Dict:
        return self.__call__(**data)

    def step_end(self, **data) -> Dict:
        return {}


class Space(Enum):
    EXP = "exp"
    LOG = "log"
