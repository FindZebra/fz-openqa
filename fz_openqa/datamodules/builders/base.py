import os
import re
from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import Optional
from typing import Union

from datasets import Dataset
from datasets import DatasetDict

from fz_openqa.datamodules.pipes import Pipe

CAMEL2SNAKE = re.compile(r"(?<!^)(?=[A-Z])")


def to_snake_format(name: str) -> str:
    return str(CAMEL2SNAKE.sub("_", name).lower())


class DatasetBuilder(ABC):
    _cache_path: Optional[str] = None
    _cache_type: Optional[Union[Dataset.__class__, DatasetDict.__class__]] = None
    _cache_dir = None

    def __init__(self, *, cache_dir: Optional[str]):
        if cache_dir is None:
            self._cache_dir = None
        else:
            self._cache_dir = os.path.join(cache_dir, to_snake_format(self.__class__.__name__))
            if not os.path.exists(self._cache_dir):
                os.makedirs(self._cache_dir)

    @abstractmethod
    def __call__(self, **kwargs) -> Union[Dataset, DatasetDict]:
        raise NotImplementedError

    def get_collate_pipe(self) -> Pipe:
        raise NotImplementedError

    def format_row(self, row: Dict[str, Any]) -> str:
        """format a row from the dataset"""
        raise NotImplementedError
