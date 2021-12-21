import json
import logging
from functools import partial
from pathlib import Path
from time import time
from typing import Any
from typing import Dict
from typing import Optional

import jsondiff
import rich
from datasets import Dataset
from datasets import DatasetDict

from fz_openqa.datamodules.index import FaissIndex
from fz_openqa.datamodules.index.index_pipes import SearchCorpus
from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.datamodules.pipes import PrintBatch
from fz_openqa.datamodules.pipes import Sequential
from fz_openqa.datamodules.utils.typing import HfDataset
from fz_openqa.utils.fingerprint import get_fingerprint

logger = logging.getLogger(__name__)


class MapWithFingerprint:
    """
    Make sure to set `new_fingerprint` to each split.
    """

    def __init__(
        self,
        pipe: Pipe,
        batched=True,
        cache_dir: str = None,
        id: str = None,
        debug: bool = False,
        **map_kwargs: Any,
    ):
        self.id = id
        self.pipe = pipe
        self.cache_dir = cache_dir
        self.debug = debug
        self.map_kwargs = {"batched": batched, **map_kwargs}

    def __call__(self, dataset: HfDataset) -> HfDataset:
        """Apply the `pipe` to the `dataset` using deterministic fingerprints."""

        if isinstance(dataset, Dataset):
            dataset = DatasetDict({"__all__": dataset})

        # check if the pipe is safe for multi-processing
        self._check_pickling(self.pipe)

        # check if the fingerprint from the previous run is identical to the current one
        if not self.map_kwargs.get("keep_in_memory", False):
            self._check_fingerprint_previous_run(dataset, name=type(self.pipe).__name__)

        # generate new fingerprints
        fingerprints = self._gen_fingerprints(dataset, self.pipe, {})

        # process dataset
        for key, dset in dataset.items():

            # adjust kwargs
            kwargs = self.map_kwargs.copy()
            if isinstance(self.pipe, SearchCorpus) and isinstance(self.pipe.index, FaissIndex):
                # FaissIndex is made to work in on thread only
                kwargs["num_proc"] = 1

            # adjust the batch size to be at least `num_proc
            kwargs["batch_size"] = max(kwargs["num_proc"], kwargs["batch_size"])

            # fingerprint
            fingerprint = fingerprints.get(key, None)
            logger.debug(f"split={key}: new_fingerprint={fingerprint}")

            pipe = self.pipe
            if self.debug:
                pipe = Sequential(
                    PrintBatch(f"{self.id} : input"), pipe, PrintBatch(f"{self.id} : output")
                )

            # process each split
            start_time = time()
            dataset[key] = dset.map(
                partial(pipe, split=key),
                new_fingerprint=fingerprint,
                with_indices=True,
                **kwargs,
            )
            logger.info(f"{self.id}, {key}: elapsed_time={time() - start_time:.2f}")

        if {"__all__"} == set(dataset.keys()):
            dataset = dataset.pop("__all__")

        return dataset

    def _check_fingerprint_previous_run(self, dataset: HfDataset, name: str):
        """Check that the pipe fingerprint is identical to the one of the previous run.
        This allows tracking down problems with on-deterministic serialization."""

        if self.cache_dir is None:
            logger.warning("cache_dir is not provided, previous fingerprints cannot be verified.")
            return

        # get a json-file from the current pipe
        fingerprints = self.pipe.fingerprint(reduce=False)
        fingerprints["__all__"] = self.pipe.fingerprint(reduce=True)

        # get the dataset fingerprint
        if isinstance(dataset, Dataset):
            dset_fingerprint = dataset._fingerprint
        elif isinstance(dataset, DatasetDict):
            dset_fingerprint = get_fingerprint({k: d._fingerprint for k, d in dataset.items()})
        else:
            raise TypeError(f"Cannot handle dataset type {type(dataset)}")

        # create the directory to store the fingerprints
        path = Path(self.cache_dir) / ".fingerprints/"
        if not path.exists():
            path.mkdir(parents=True)

        # compare to previously saved fingerprints
        file = path / f"{name}-{dset_fingerprint}.json"
        if file.exists():
            prev_fingerprints = json.loads(file.read_text())
            diff = jsondiff.diff(prev_fingerprints, fingerprints)
            if len(diff) > 0:
                logger.warning(
                    f"Fingerprint for {name} changed from the latest run. "
                    f"Caching cannot be used. Enable debug logging mode to see the diff."
                )
                logger.debug(f"Fingerprint diff={diff}")
                if self.debug:
                    rich.print(f"[magenta] {name}: Fingerprints are different !")
                    rich.print(diff)

        file.write_text(json.dumps(fingerprints, indent=2))

    def _check_pickling(self, pipe: Pipe):
        """check that the pipe can be pickled, which is necessary for multiprocessing"""
        try:
            if not pipe.dill_inspect(reduce=True):
                rich.print(pipe.dill_inspect())
                raise TypeError(
                    "Couldn't pickle pipe. Code would fail if `num_proc`>1. "
                    f"Make sure the pipe {pipe} can be pickled."
                )
        except Exception as e:
            rich.print("=== pipe pickle ===")
            rich.print(pipe.dill_inspect(reduce=False))
            raise e

    def _gen_fingerprints(self, dataset: DatasetDict, pipe: Pipe, params: Optional[Dict]):
        params = params or {}

        return {
            k: get_fingerprint(
                {
                    "dataset": dset._fingerprint,
                    "pipe": pipe.fingerprint(reduce=True),
                    "params": params,
                }
            )
            for k, dset in dataset.items()
        }
