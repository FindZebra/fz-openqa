from typing import Dict
from typing import Optional

import rich
from datasets import Dataset
from datasets import DatasetDict
from datasets.fingerprint import update_fingerprint

from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.datamodules.utils.typing import HfDataset


class MapWithFingerprint:
    """
    Make sure to set `new_fingerprint` to each split.
    """

    def __init__(
        self, pipe: Pipe, batched=True, _id: str = None, **map_kwargs
    ):
        self._id = _id
        self.pipe = pipe
        self.map_kwargs = {"batched": batched, **map_kwargs}

    def __call__(self, dataset: HfDataset) -> HfDataset:
        """Apply the `pipe` to the `dataset` using deterministic fingerprints."""

        if isinstance(dataset, Dataset):
            dataset = DatasetDict({"__all__": dataset})

        # check if the pipe is safe for multiprocesssing
        self._check_pickling(self.pipe)

        # generate new fingerprints
        fingerprints = self._gen_fingerprints(dataset, self.pipe, {})

        # process dataset
        for key, dset in dataset.items():
            dataset[key] = dset.map(
                self.pipe,
                new_fingerprint=fingerprints.get(key, None),
                **self.map_kwargs,
            )

        if {"__all__"} == set(dataset.keys()):
            dataset = dataset.pop("__all__")

        return dataset

    def _check_pickling(self, pipe: Pipe):
        """check that the pipe can be pickled, which is necessary for multiprocessing"""
        if not pipe.dill_inspect(reduce=True):
            rich.print(pipe.dill_inspect())
            raise TypeError(
                "Couldn't pickle pipe. Code would fail if `num_proc`>1. "
                f"Make sure the pipe {pipe} can be pickled."
            )

    def _gen_fingerprints(
        self, dataset: DatasetDict, pipe: Pipe, params: Optional[Dict]
    ):
        params = params or {}
        return {
            k: update_fingerprint(
                dset._fingerprint,
                pipe,
                params,
            )
            for k, dset in dataset.items()
        }
