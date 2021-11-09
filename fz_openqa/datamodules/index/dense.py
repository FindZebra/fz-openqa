import logging
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional

import dill
import faiss
import numpy as np
import pytorch_lightning
from datasets import Dataset
from faiss.swigfaiss import Index as FaissSwigIndex
from pytorch_lightning import Trainer
from rich.progress import track
from torch.utils.data import DataLoader

from fz_openqa.callbacks.store_results import StoreResultCallback
from fz_openqa.datamodules.index.base import Index
from fz_openqa.datamodules.index.base import SearchResult
from fz_openqa.datamodules.pipes import Collate
from fz_openqa.datamodules.pipes import FilterKeys
from fz_openqa.datamodules.pipes import Forward
from fz_openqa.datamodules.pipes import Parallel
from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.datamodules.pipes import RenameKeys
from fz_openqa.datamodules.pipes import Sequential
from fz_openqa.datamodules.pipes import ToNumpy
from fz_openqa.datamodules.pipes.control.filter_keys import KeyIn
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.fingerprint import get_fingerprint

logger = logging.getLogger(__name__)


class FaissIndex(Index):
    """Dense indexing using faiss"""

    vectors_column_name = "__vectors__"
    _index: FaissSwigIndex = None
    model: Callable = None

    def __init__(
        self,
        dataset: Dataset,
        *,
        model: Callable,
        trainer: Optional[Trainer] = None,
        nlist: int = 10,
        index_key: str = "document.row_idx",
        model_output_keys: List[str],
        batch_size: int = 10,
        num_workers: int = 0,
        collate_pipe: Pipe = Collate(),
        metric_type: str = faiss.METRIC_INNER_PRODUCT,
        **kwargs,
    ):

        #  save fingerprints
        self._model_fingerprint = get_fingerprint(model)
        self._datadset_fingerprint = dataset._fingerprint

        # model and params
        self.model = model
        self.nlist = nlist
        self.metric_type = metric_type

        self.index_key = index_key
        self.model_output_keys = model_output_keys

        # trainer and dataloader
        self.trainer = trainer
        self.batch_size = batch_size
        self.num_workers = num_workers

        # collate pipe use to convert dataset rows into a batch
        self.collate = collate_pipe

        # process pipe, used to process a batch with the model
        # warning: this is actually not used when using the trainer
        self.process = Sequential(
            Parallel(Forward(model=model), FilterKeys(KeyIn([self.index_key]))), ToNumpy()
        )

        # postprocessing: rename the model outputs `model_output_keys` to `vectors_column_name`
        self.postprocess = self.get_rename_output_names_pipe(
            inputs=self.model_output_keys, output=self.vectors_column_name
        )

        # call the super: build the index
        super(FaissIndex, self).__init__(dataset=dataset, **kwargs)

    def build(self, dataset: Dataset, **kwargs):
        """Index a dataset."""

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate,
            shuffle=False,
        )

        # process batches
        processed_batches = self._process_batches(loader, trainer=self.trainer)

        # build an iterator over the processed batches
        it = iter(processed_batches)

        # init faiss index and add the 1st batch
        batch = next(it)
        self._init_index(batch)
        self._add_batch_to_index(batch)

        # iterate through the remaining batches and add them to the index
        while True:
            try:
                batch = next(it)
                self._add_batch_to_index(batch)
            except Exception:
                break

        logger.info(
            f"Index is_trained={self._index.is_trained}, "
            f"size={self._index.ntotal}, type={type(self._index)}"
        )

    def _process_batches(
        self, loader: DataLoader, trainer: Optional[Trainer] = None, progress_bar: bool = True
    ) -> Iterable[Batch]:

        """
        This method iterates over batches, which for each batch:
        1. process the batch using the model
        2. postprocess the output of the model (renaming keys, filtering keys, etc)

        This function processes batches as they are loaded from the dataloader.
        """

        if trainer is not None and isinstance(self.model, pytorch_lightning.LightningModule):
            # retrieve the callback used to store the results
            store_results_callback = self._get_store_results_callback(self.trainer)

            # run the trainer predict method, model.forward() is called
            # for each batch and store into the callback cache
            trainer.predict(model=self.model, dataloaders=loader, return_predictions=False)

            # return an iterator over the cached batches
            output_batches = store_results_callback.iter_batches()
        else:
            if progress_bar:
                loader = track(iter(loader), description="Processing dataset")

            # process each batch sequentially using `self.process`
            output_batches = map(self.process, loader)

        final_outputs = map(self.postprocess, output_batches)
        for batch in final_outputs:
            yield batch

    def _get_store_results_callback(self, trainer: Trainer):
        store_results_callback = [
            c for c in trainer.callbacks if isinstance(c, StoreResultCallback)
        ]
        msg = f"Callback of type {type(StoreResultCallback)} must be provided."
        assert len(store_results_callback) == 1, msg
        store_results_callback = store_results_callback[0]
        return store_results_callback

    def dill_inspect(self) -> Dict[str, bool]:
        """check if the module can be pickled."""
        return {
            "__all__": dill.pickles(self),
            **{k: dill.pickles(v) for k, v in self.__getstate__().items()},
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_index"] = faiss.serialize_index(state["_index"])
        return state

    def __setstate__(self, state):
        state["_index"] = faiss.deserialize_index(state["_index"])
        self.__dict__.update(state)

    def _init_index(self, batch):
        vectors = batch[self.vectors_column_name]
        assert len(vectors.shape) == 2
        self._index = faiss.IndexFlat(vectors.shape[-1], self.metric_type)

    def _add_batch_to_index(self, batch: Batch, dtype=np.float32):
        # check indexes
        indexes = batch[self.index_key]
        msg = f"Indexes are not contiguous (i.e. 1, 2, 3, 4),\nindexes={indexes}"
        assert all(indexes[:-1] + 1 == indexes[1:]), msg
        msg = (
            f"The stored index and the indexes are not contiguous, "
            f"\nindex_size={self._index.ntotal}, first_index={indexes[0]}"
        )
        assert self._index.ntotal == indexes[0], msg

        # add the vectors
        vector: np.ndarray = batch[self.vectors_column_name]
        assert isinstance(vector, np.ndarray), f"vector {type(vector)} is not a numpy array"
        assert len(vector.shape) == 2, f"{vector} is not a 2D array"
        vector = vector.astype(dtype)
        self._index.add(vector)

    @property
    def is_indexed(self):
        return self._index is None or self._index.is_trained

    def search(
        self,
        query: Batch,
        *,
        k: int = 1,
        **kwargs,
    ) -> SearchResult:
        """Search the index using the `query` and
        return the index of the results within the original dataset."""
        query = self.process(query)
        query = self.postprocess(query)
        vectors = query[self.vectors_column_name]
        score, indices = self._index.search(vectors, k)
        return SearchResult(score=score, index=indices)

    def get_rename_output_names_pipe(self, inputs: List[str], output: str) -> Pipe:
        """Format the output of the model"""
        return Sequential(
            RenameKeys({key: output for key in inputs}),
            FilterKeys(KeyIn([output, self.index_key])),
        )
