import tempfile
from typing import Iterable
from typing import List
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytorch_lightning as pl
from pytorch_lightning import Callback

from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.functional import cast_values_to_numpy
from fz_openqa.utils.functional import iter_batch_rows


class StorePredictionsCallback(Callback):
    """Allows storing the output of each `prediction_step` into a `pyarrow` table.
    The Table can be access using the attribute `data` or using the method `iter_batches()`"""

    _table: Optional[pa.Table] = None
    _writer: Optional[pq.ParquetWriter] = None

    def __init__(self, cache_dir: Optional[str] = None, store_fields: Optional[List[str]] = None):
        self.cache_dir = cache_dir
        if store_fields is None:
            store_fields = []
        self.store_fields = store_fields
        self._reset()

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
        batch = {k: v for k, v in batch.items() if k in self.store_fields}
        self._append_batch_to_cache({**batch, **outputs})

    def _reset(self):
        """get a new temporary cache file and close any existing writer."""
        self._close_writer()
        self.cache = tempfile.TemporaryFile(dir=self.cache_dir)

    def _close_writer(self):
        """close any existing writer"""
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def _append_table_to_cache(self, table: pa.Table) -> None:
        """write a table to file."""
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.cache, table.schema)
        self._writer.write_table(table)

    def _append_batch_to_cache(self, batch: Batch) -> None:
        """Append the batch to the cached `pyarrow.Table`"""
        batch = cast_values_to_numpy(batch, as_contiguous=False)
        df = pd.DataFrame(iter_batch_rows(batch))
        # Convert from pandas to Arrow
        table = pa.Table.from_pandas(df)
        # write to the cache file
        self._append_table_to_cache(table)

    @property
    def data(self) -> pa.Table:
        """Returns the cached `pyarrow.Table`"""
        return pq.read_table(self.cache)

    def iter_batches(self, batch_size=1000) -> Iterable[Batch]:
        """Returns an iterator over the cached batches"""
        for batch in self.data.to_batches(max_chunksize=batch_size):
            yield cast_values_to_numpy(batch.to_pydict())
