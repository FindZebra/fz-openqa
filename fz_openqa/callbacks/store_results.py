import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytorch_lightning as pl
import rich
from pytorch_lightning import Callback

from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.functional import cast_values_to_numpy
from fz_openqa.utils.functional import iter_batch_rows

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
        cache_dir: Optional[str] = None,
        store_fields: Optional[List[str]] = None,
        cache_name: Optional[str] = None,
        persist: bool = False,
    ):
        if cache_dir is None:
            # todo: set proper default cache path
            cache_dir = ".cache/fz-openqa/"

        self.cache_dir = Path(cache_dir) / "predictions"

        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True)

        if persist is False:
            raise NotImplementedError
            self.cache_dir = tempfile.TemporaryDirectory(dir=self.cache_dir)
            rich.print(f">> cache_dir={self.cache_dir}")

        if cache_name is None:
            cache_name = uuid.uuid4().hex
        self.cache_name = f"{cache_name}.arrow"

        if store_fields is None:
            store_fields = []
        self.store_fields = list(set(store_fields + [IDX_COL]))
        self._reset()
        logger.info(f"Storing predictions to {self.cache_file}")
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
        if self.cache_name is None:
            self.cache_file = tempfile.TemporaryFile(dir=self.cache_dir)
        else:
            self.cache_file = os.path.join(self.cache_dir, self.cache_name)

    @property
    def is_written(self):
        if self.cache_name is None:
            return False
        else:
            return os.path.exists(self.cache_file)

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
        batch = cast_values_to_numpy(batch, as_contiguous=False)
        df = pd.DataFrame(iter_batch_rows(batch))
        # Convert from pandas to Arrow
        table = pa.Table.from_pandas(df)
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
