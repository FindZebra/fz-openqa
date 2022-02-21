from __future__ import annotations

import abc
import json
import logging
from copy import copy
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import torch
from hydra.utils import instantiate

from fz_openqa.datamodules.component import Component
from fz_openqa.utils.datastruct import PathLike
from fz_openqa.utils.tensor_arrow import TensorArrowTable

logger = logging.getLogger(__name__)


class IndexHandler(Component):
    """This class implements handles a dense index."""

    __metaclass__ = abc.ABCMeta

    def __init__(self, *, path: PathLike, **kwargs):
        super(IndexHandler, self).__init__()
        self.path: Path = Path(path)
        self.config = copy(kwargs)

    @classmethod
    def load_from_path(cls, path: PathLike):
        config_path = path / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)
            instance = instantiate(config)
            instance.load()

        return instance

    def build(self, vectors: torch.Tensor | TensorArrowTable):
        if self.exists():
            logger.info(f"Loading index from {self.path}")
            self.load()
        else:
            logger.info(f"Creating index at {self.path}")
            self._build(vectors, **copy(self.config))
            assert self.exists(), f"Index {type(self).__name__} was not created."
            self.save()

    def rm(self):
        """Remove the index."""
        if self.path.exists():
            if self.path.is_dir():
                self.path.rmdir()
            else:
                self.path.unlink()

    def exists(self):
        """Check if the index exists."""
        return self.path.exists()

    @abc.abstractmethod
    def __len__(self) -> int:
        """Return the number of vectors in the index."""
        ...

    @abc.abstractmethod
    def _build(self, vectors: torch.Tensor, **kwargs):
        """build the index from the vectors."""
        ...

    def save(self):
        """save the index to file"""
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(str(self.config_file), "w") as f:
            f.write(json.dumps(self._get_full_config()))

    def load(self):
        """save the index to file"""
        with open(str(self.config_file), "r") as f:
            self.config = json.load(f)
            self.config.pop("_target_")
            self.config.pop("path")

    @property
    def config_file(self) -> Path:
        return self.path / "config.json"

    def _get_full_config(self) -> Dict[str, Any]:
        cfg = copy(self.config)
        cfg["path"] = str(self.path)
        cfg["_target_"] = type(self).__module__ + "." + type(self).__qualname__
        return cfg

    @abc.abstractmethod
    def cpu(self):
        """Move the index to CPU."""
        ...

    @abc.abstractmethod
    def cuda(self, devices: Optional[List[int]] = None):
        """Move the index to CUDA."""
        ...

    @abc.abstractmethod
    def free_memory(self):
        """Free the memory of the index."""
        ...

    @property
    @abc.abstractmethod
    def is_up(self) -> bool:
        """Check if the index is up."""
        ...

    @abc.abstractmethod
    def __call__(
        self, query: torch.Tensor, *, k: int, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Search the index."""
        ...
