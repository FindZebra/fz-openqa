import logging
import uuid
from pathlib import Path
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytorch_lightning as pl
from datasets.features import numpy_to_pyarrow_listarray
from pytorch_lightning import Callback

from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.datastruct import PathLike
from fz_openqa.utils.functional import cast_values_to_numpy

logger = logging.getLogger(__name__)

IDX_COL = "__idx__"


class StorePredictionsCallback(Callback):
    """Allows storing the output of each `prediction_step` into a `pyarrow` table.
    The Table can be access using the attribute `table` or using the method `iter_batches()`"""

    _table: Optional[pa.Table] = None
    _writer: Optional[pq.ParquetWriter] = None
    cache_file: Optional[Union[str, Path]] = None

    def __init__(
        self,
        store_fields: Optional[List[str]] = None,
        cache_file: Optional[PathLike] = None,
    ):
        if cache_file is None:
            cache_file = uuid.uuid4().hex
            cache_file = f"{cache_file}.arrow"

        self.cache_file = Path(cache_file)
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)

        if store_fields is None:
            store_fields = []
        self.store_fields = list(set(store_fields + [IDX_COL]))
        self._reset()
        logger.info(f"Initialized {type(self).__name__} with cache_file={self.cache_file}")
        logger.info(f"is_written={self.is_written}")

    def on_predict_epoch_end(self, *args, **kwargs) -> None:
        self._close_writer()

    def on_predict_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self._reset()

    def on_predict_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Batch,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """store the outputs of the prediction step to the cache"""
        msg = f"{self.store_fields} not in batch (keys={batch.keys()})"
        assert all(f in batch for f in self.store_fields), msg
        batch = {k: v for k, v in batch.items() if k in self.store_fields}
        self._append_batch_to_cache({**batch, **outputs})

    def _reset(self):
        """get a new temporary cache file and close any existing writer."""
        self._close_writer()

    @property
    def is_written(self):
        return self.cache_file.exists()

    def _close_writer(self):
        """close any existing writer"""
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def _append_table_to_cache(self, table: pa.Table) -> None:
        """write a table to file."""
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.cache_file, table.schema)
        self._writer.write_table(table)

    def _append_batch_to_cache(self, batch: Batch) -> None:
        """Append the batch to the cached `pyarrow.Table`"""
        batch = cast_values_to_numpy(batch, as_contiguous=False, dtype=np.float32)

        def cast_values(x: np.ndarray):
            return numpy_to_pyarrow_listarray(x)

        # build the pyarrow table
        batch = {k: cast_values(v) for k, v in batch.items()}
        table = pa.table(batch)

        # write to the cache file
        self._append_table_to_cache(table)

    @property
    def table(self) -> pa.Table:
        """Returns the cached `pyarrow.Table`"""
        if self._table is None:
            self._table = pq.read_table(self.cache_file)
        return self._table

    def __getitem__(self, item) -> Dict:
        return self.table[item]

    def iter_batches(self, batch_size=1000) -> Iterable[Batch]:
        """Returns an iterator over the cached batches"""
        for batch in self.table.to_batches(max_chunksize=batch_size):
            yield cast_values_to_numpy(batch.to_pydict())

    def __del__(self):
        self._close_writer()
        if self._table is not None:
            del self._table
