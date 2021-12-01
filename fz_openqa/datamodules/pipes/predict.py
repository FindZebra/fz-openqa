import functools
import logging
import os.path
from enum import Enum
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sized
from typing import Tuple
from typing import Union

import pyarrow as pa
import pyarrow.parquet as pq
import pytorch_lightning as pl
import rich
import torch
from datasets import Dataset
from datasets import DatasetDict
from datasets import Split
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import move_data_to_device
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import SequentialSampler

from fz_openqa.callbacks.store_results import IDX_COL
from fz_openqa.callbacks.store_results import StorePredictionsCallback
from fz_openqa.datamodules.pipes import Collate
from fz_openqa.datamodules.pipes import Parallel
from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.fingerprint import get_fingerprint
from fz_openqa.utils.functional import cast_values_to_numpy
from fz_openqa.utils.functional import cast_values_to_torch
from fz_openqa.utils.functional import is_index_contiguous

logger = logging.getLogger(__name__)

DEFAULT_LOADER_KWARGS = {"batch_size": 10, "num_workers": 2, "pin_memory": True}
CACHE_FILE = Union[Path, str]


class OutputFormat(Enum):
    """
    Enum for the output format of the predictions.
    """

    NUMPY = "numpy"
    TORCH = "torch"


class AddRowIdx(TorchDataset):
    """This class is used to add the column `IDX_COL` to the batch"""

    def __init__(self, dataset: Sized):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item) -> Batch:
        batch = self.dataset[item]
        batch[IDX_COL] = item
        return batch


class Predict(Pipe):
    """Allow computing predictions for a dataset using Lightning.
    This pipes requires first to call `cache()` on the target dataset.
    The predictions are then computed and stored to a cache file.
    Once cached, the pipe can be called on the dataset again to
    and the predictions are load from the cache

    Notes
    -----
    This class can handle both Dataset and DatasetDict. However, the `idx` kwarg
    must be passed to __call__ to successfully read the cache. This can be done using
    the argument `with_indices` in `Dataset.map()` or `DatasetDict.map()`.

    When using a `DatasetDict`, the kwarg `split` must be set to read the cache.


    Attributes
    ----------
    cache_file : Union[Path, str, tempfile.TemporaryFile]
        The cache file(s) to store the predictions.
        A dictionary is used when processing multiple splits.

    Examples
    ---------
    1. Using Predict on a Dataset:

    ```python
    predict = Predict(model=model)

    # cache the dataset: Dataset
    predict.cache(dataset,
                  trainer=trainer,
                  collate_fn=collate_fn,
                  cache_dir=cache_dir,
                  loader_kwargs={'batch_size': 2},
                  persist=True)

    dataset = dataset.map(dataset,
                          batched=True,
                          batch_size=5,
                          with_indices=True)
    ```

    2. Using Predict on a DatasetDict:

    ```python
    predict = Predict(model=model)

    # cache the dataset: DatasetDict
    predict.cache(dataset,
                  trainer=trainer,
                  collate_fn=collate_fn,
                  cache_dir=cache_dir,
                  loader_kwargs={'batch_size': 2},
                  persist=True)

    dataset = DatasetDict({split: d.map(partial(predict, split=split),
                                        batched=True,
                                        batch_size=5,
                                        with_indices=True)
                           for split, d in dataset.items()})
    ```

    """

    model: Optional[pl.LightningModule] = None
    cache_file: Optional[Union[CACHE_FILE, Dict[Split, CACHE_FILE]]] = None
    _loaded_table: Optional[pa.Table] = None
    _loaded_split: Optional[Split] = None
    _pickle_exclude = ["model", "_loaded_table", "_loaded_split"]

    def __init__(self, model: pl.LightningModule, requires_cache: bool = False, **kwargs):
        """
        Parameters
        ----------
        model
            The model to use for predictions.
        requires_cache
            If True, the cache file must be set before calling `__call__`.
        kwargs
            Additional keyword arguments passed to `Pipe`.
        """
        super(Predict, self).__init__(**kwargs)
        self.model = model
        self.requires_cache = requires_cache

    def invalidate_cache(self):
        """Reset the cache"""
        self._loaded_table = None
        self.cache_file = None
        self._loaded_split = None

    @torch.no_grad()
    def _call_batch(
        self,
        batch: Batch,
        idx: List[int] = None,
        split: Optional[Split] = None,
        format: OutputFormat = OutputFormat.NUMPY,
        **kwargs,
    ) -> Batch:
        """
        Call the model on the batch or read the cached predictions.

        Parameters
        ----------
        batch
            The batch to process.
        idx
            The indices of the examples to process.
        split
            The split to process (Optional)
        format
            The format to return the predictions in.
        kwargs
            Additional keyword arguments passed to the model when not using the cache.
        Returns
        Batch
            The output of the model (as numpy arrays).
        -------

        """
        use_cache = self.cache_file is not None
        if use_cache and idx is None:
            logger.warning(
                "Cache file exists but `idx` is not set, the cache won't be used. "
                "Make sure to call `Dataset.map` with `use_indices=True`"
            )
            use_cache = False
        if use_cache:
            return self._process_batch_with_cache(batch, idx=idx, split=split, format=format)
        else:
            if self.requires_cache:
                raise ValueError(
                    "This pipe explicitly requires calling "
                    "`pipe.cache()` before subsequent uses."
                )
            return self._process_batch_without_cache(batch, format=format, **kwargs)

    def _process_batch_with_cache(
        self, _: Optional[Batch], idx: List[int], split: Optional[Split], format: OutputFormat
    ) -> Batch:
        """Process a batch using the cache file"""
        table = self.read_table(split)
        if is_index_contiguous(idx):
            msg = "The table is smaller than the max. of the input index."
            assert table.num_rows > max(idx), msg
            rows = table[min(idx) : max(idx) + 1]

        else:
            rows = table.take(idx)

        if format == OutputFormat.NUMPY:
            return cast_values_to_numpy(rows.to_pydict())
        elif format == OutputFormat.TORCH:
            return cast_values_to_torch(rows.to_pydict())
        else:
            raise ValueError(f"Unknown format: {format}")

    def _process_batch_without_cache(
        self, batch: Batch, format: OutputFormat = OutputFormat.NUMPY, **kwargs
    ) -> Batch:
        """Process the batch using the model and without the cache"""
        if isinstance(self.model, nn.Module):
            device = next(iter(self.model.parameters())).device
            batch = move_data_to_device(batch, device)
        # process with the model (Dense or Sparse)
        output = self.model(batch, **kwargs)

        if format == OutputFormat.NUMPY:
            return cast_values_to_numpy(output)
        elif format == OutputFormat.TORCH:
            return cast_values_to_torch(output)
        else:
            raise ValueError(f"Unknown format: {format}")

    @functools.singledispatchmethod
    @torch.no_grad()
    def cache(
        self,
        dataset: Dataset,
        *,
        trainer: Optional[Trainer] = None,
        collate_fn: Optional[Pipe] = None,
        loader_kwargs: Optional[Dict] = None,
        cache_dir: Optional[str] = None,
        split: Optional[Split] = None,
        persist: bool = True,
    ) -> CACHE_FILE:
        """
        Cache the predictions of the model on the dataset.

        Parameters
        ----------
        dataset
            The dataset to cache the predictions on.
        trainer
            The pytorch_lightning Trainer used to accelerate the prediction
        collate_fn
            The collate_fn used to process the dataset (e.g. builder.get_collate_pipe())
        loader_kwargs
            The keyword arguments passed to the DataLoader
        cache_dir
            The directory to store the cache file(s)
        split
            (Optional) The split to cache the predictions on. Leave to None when using a Dataset.
        persist
            (Optional) Whether to persist the cache file(s) for subsequent runs.
            If set to False, the cache file is deleted when the session ends (tempfile).
        Returns
        Union[Path, str, tempfile.TemporaryFile]
            The path to the cache file
        -------

        """
        # check the setup of the model and trainer
        assert trainer is not None, "Trainer must be provided to process batches."
        msg = f"Model must be a LightningModule, got {type(self.model)}"
        assert isinstance(self.model, LightningModule), msg

        # define a unique fingerprint for the cache file
        fingerprint = get_fingerprint(
            {"model": get_fingerprint(self.model), "split": split, "dataset": dataset._fingerprint}
        )

        # init a callback to store predictions and add it to the Trainer
        callback = StorePredictionsCallback(
            cache_name=fingerprint, cache_dir=cache_dir, persist=persist
        )

        # define a collate_fn to process the dataset
        if callback.is_written:
            logger.info(f"Loading pre-computed vectors from {callback.cache_file}")
            cached_table = self.read_table_from_cache_file(callback.cache_file)
            if not len(cached_table) == len(dataset):
                raise ValueError(
                    f"Dataset of length={len(dataset)} not matching "
                    f"the cache with length={len(cached_table)}. "
                    f"Consider deleting and re-computing these vectors.\n"
                    f"path={os.path.abspath(callback.cache_file)}"
                )
        else:
            # process the whole dataset using Trainer
            self._process(
                dataset,
                callback=callback,
                trainer=trainer,
                collate_fn=collate_fn,
                loader_kwargs=loader_kwargs,
            )

        # Retrieve the cache file
        cache_file = callback.cache_file

        # store the `cache_file` as an attribute
        if split is None:
            self.cache_file = cache_file
        else:
            if self.cache_file is None:
                self.cache_file = {}
            self.cache_file[split] = cache_file

        return cache_file

    @cache.register(DatasetDict)
    def _(self, dataset: DatasetDict, **kwargs) -> Dict[Split, CACHE_FILE]:
        """
        Cache the predictions of the model for a DatasetDict.

        Parameters
        ----------
        dataset
            The DatasetDict to cache the predictions on.
        kwargs
            Additional keyword arguments passed to `cache`.
        Returns
        Dict[str, Union[Path, str, tempfile.TemporaryFile]]
            The path to the cache file for each split
        -------
        """
        return {split: self.cache(dset, split=split, **kwargs) for split, dset in dataset.items()}

    def _process(
        self,
        dataset: Dataset,
        *,
        callback: StorePredictionsCallback,
        trainer: Optional[Trainer],
        collate_fn: Optional[Callable],
        loader_kwargs: Optional[Dict],
    ) -> CACHE_FILE:
        """
        Process the dataset using the Trainer an the model.
        """
        trainer.callbacks.append(callback)

        # init the dataloader (the collate_fn and dataset are wrapped to return the ROW_IDX)
        loader = self.init_loader(dataset, collate_fn=collate_fn, loader_kwargs=loader_kwargs)

        # modify the dataloader to return __idx__ aside from the original data
        loader = self._wrap_dataloader(loader)
        assert isinstance(
            loader.sampler, SequentialSampler
        ), "Cannot handle DataLoader with shuffle=True."

        # run the trainer predict method, model.forward() is called
        # for each batch and store into the callback cache
        trainer.predict(model=self.model, dataloaders=loader, return_predictions=False)

        cache_file = callback.cache_file
        trainer.callbacks.remove(callback)
        return cache_file

    def read_table(self, split: Optional[Split]) -> pa.Table:
        """Returns the cached `pyarrow.Table` for the given split"""

        if isinstance(self.cache_file, dict):
            assert split is not None, "Split must be provided when using a dict of cache files."

        if split is None:
            assert not isinstance(
                self.cache_file, dict
            ), "Split must be provided to access the table."
            if self._loaded_table is None:
                self._loaded_table = self.read_table_from_cache_file(self.cache_file)
        elif split == self._loaded_split and self._loaded_table is not None:
            pass
        else:
            cache = self.cache_file[split]
            self._loaded_table = self.read_table_from_cache_file(cache)
            self._loaded_split = split

        return self._loaded_table

    @staticmethod
    def read_table_from_cache_file(cache_file: str) -> pa.Table:
        read_args = {"memory_map": True}
        return pq.read_table(cache_file, **read_args)

    @staticmethod
    def init_loader(
        dataset: Dataset,
        collate_fn: Optional[Callable] = None,
        loader_kwargs: Optional[Dict] = None,
        wrap_indices: bool = True,
    ) -> DataLoader:
        """Initialize the dataloader.

        if `wrap_indices` is True, the dataset and collate_fn are wrapped
        such that each example comes with a valid `ROW_IDX` value.

        Parameters
        ----------
        dataset
            The dataset to initialize the dataloader for.
        collate_fn
            The collate_fn to use for the dataloader.
        loader_kwargs
            Additional keyword arguments to pass to the dataloader.
        wrap_indices
            Whether to wrap the dataset and collate_fn with a valid `ROW_IDX` value.

        Returns
        -------
        DataLoader
            The initialized dataloader.
        """
        if collate_fn is None:
            collate_fn = Collate()

        if wrap_indices:
            dataset = Predict._wrap_dataset(dataset)
            collate_fn = Predict._wrap_collate_fn(collate_fn)

        loader_kwargs = loader_kwargs or DEFAULT_LOADER_KWARGS
        loader = DataLoader(
            dataset,
            collate_fn=collate_fn,
            shuffle=False,
            **loader_kwargs,
        )
        return loader

    @staticmethod
    def _wrap_collate_fn(collate_fn: Callable) -> Pipe:
        """Wrap the collate_fn to return IDX_COL along the batch values"""
        return Parallel(collate_fn, Collate(IDX_COL))

    @staticmethod
    def _wrap_dataloader(loader: DataLoader) -> DataLoader:
        """
        Return a copy of the original dataset such that the `dataset` and `collate_fn` are
        updated such as to return the field `ROW_IDX` along the original data.
        """
        if isinstance(loader.dataset, AddRowIdx):
            return loader

        def exclude(k):
            excluded_values = ["dataset", "collate_fn", "sampler", "batch_sampler"]
            return k in excluded_values or k.startswith("_")

        args = {k: v for k, v in loader.__dict__.items() if not exclude(k)}
        return DataLoader(
            dataset=Predict._wrap_dataset(loader.dataset),
            collate_fn=Predict._wrap_collate_fn(loader.collate_fn),
            **args,
        )

    @staticmethod
    def _wrap_dataset(dataset: Dataset) -> TorchDataset:
        """Wrap the dataset to return IDX_COL along the batch values"""
        return AddRowIdx(dataset)

    def __getstate__(self) -> Dict:
        state = {k: v for k, v in self.__dict__.items() if k not in self._pickle_exclude}
        return state.copy()

    def __setstate__(self, state: Dict):
        self.__dict__.update(state)
        for k in self._pickle_exclude:
            self.__dict__[k] = None

    def to_json_struct(self, exclude: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """
        Override `to_json_struct` to exclude the cache location from the fingerprint.
        # todo: unsure about the efficiency of this.
          if fingerprinting is not working as expected, this may be the cause.

        Warnings
        --------
        exclude list will also be pass to the attribute pipes.
        """

        exclude = exclude or []
        exclude += ["cache_file", "cache_dir"]
        super(Predict, self).to_json_struct(exclude=exclude, **kwargs)
