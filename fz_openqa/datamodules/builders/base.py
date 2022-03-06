from __future__ import annotations

import os
import re
from abc import ABCMeta
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from datasets import Dataset
from datasets import DatasetDict
from omegaconf import DictConfig
from omegaconf import OmegaConf

from fz_openqa.datamodules.analytics.base import Analytic
from fz_openqa.datamodules.pipes import Pipe

CAMEL2SNAKE = re.compile(r"(?<!^)(?=[A-Z])")


def to_snake_format(name: str) -> str:
    """convert a class name (Camel style) to file style (Snake style)"""
    return str(CAMEL2SNAKE.sub("_", name).lower())


class DatasetBuilder:
    """
    DatasetBuilder is a class that is responsible for building a dataset.
    """

    __metaclass__ = ABCMeta
    _cache_path: Optional[str] = None
    _cache_type: Optional[Union[Dataset.__class__, DatasetDict.__class__]] = None
    _cache_dir = None

    def __init__(
        self,
        *,
        cache_dir: Optional[str],
        analytics: Optional[Analytic | List[Analytic], Dict[str, Analytic]] = None,
    ):
        if cache_dir is None:
            self._cache_dir = None
        else:
            self._cache_dir = os.path.join(cache_dir, to_snake_format(self.__class__.__name__))
            if not os.path.exists(self._cache_dir):
                os.makedirs(self._cache_dir)

        if isinstance(analytics, DictConfig):
            analytics = OmegaConf.to_object(analytics)
        self.analytics = analytics

    def __call__(self, *args, **kwargs):
        dataset = self._call(*args, **kwargs)

        self.run_analytics(dataset)

        return dataset

    def run_analytics(self, dataset):
        if self.analytics is None:
            return
        elif isinstance(self.analytics, list):
            for a in self.analytics:
                a(dataset, builder=self)
        elif isinstance(self.analytics, dict):
            for k, v in self.analytics.items():
                v(dataset, builder=self)
        elif isinstance(self.analytics, Analytic):
            self.analytics(dataset, builder=self)
        else:
            raise TypeError(
                f"Analytics must be a list, dict or Analytic, " f"got {type(self.analytics)}"
            )

    @abstractmethod
    def _call(self, *args, **kwargs) -> Union[Dataset, DatasetDict]:
        raise NotImplementedError

    def get_collate_pipe(self) -> Pipe:
        raise NotImplementedError

    def format_row(self, row: Dict[str, Any], **kwargs) -> str:
        """format a row from the dataset

        Parameters
        ----------
        **kwargs
        """
        raise NotImplementedError
