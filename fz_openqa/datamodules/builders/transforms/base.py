from __future__ import annotations

import abc
from typing import Optional

from datasets import Dataset
from datasets import DatasetDict

from fz_openqa.datamodules.utils.datastruct import OpenQaConfig
from fz_openqa.datamodules.utils.datastruct import OpenQaDataset
from fz_openqa.datamodules.utils.typing import HfDataset


class DatasetTransform:
    @abc.abstractmethod
    def __call__(self, dataset: HfDataset, **kwargs) -> HfDataset:
        ...


class OpenQaTransform:
    def __call__(
        self,
        dataset_or_config: DatasetDict | OpenQaConfig,
        openqa_config: Optional[OpenQaConfig] = None,
        **kwargs,
    ) -> OpenQaDataset | OpenQaConfig:
        if isinstance(dataset_or_config, (Dataset, DatasetDict)):
            return self._transform_dataset(dataset_or_config, openqa_config, **kwargs)
        elif isinstance(dataset_or_config, OpenQaConfig):
            return self._transform_config(dataset_or_config, **kwargs)
        else:
            raise TypeError(f"Unsupported type {type(dataset_or_config)}")

    @abc.abstractmethod
    def _transform_dataset(
        self, dataset: HfDataset, openqa_config: Optional[OpenQaConfig], **kwargs
    ) -> OpenQaDataset:
        ...

    @abc.abstractmethod
    def _transform_config(self, openqa_config: OpenQaConfig, **kwargs) -> OpenQaConfig:
        ...
