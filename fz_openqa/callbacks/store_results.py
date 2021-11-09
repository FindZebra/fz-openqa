import tempfile
from typing import Any
from typing import Dict
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
from fz_openqa.utils.functional import get_batch_eg
from fz_openqa.utils.functional import infer_batch_size
from fz_openqa.utils.pretty import pprint_batch


def iter_batch_rows(batch: Batch) -> Iterable[Dict]:
    batch_size = infer_batch_size(batch)
    for i in range(batch_size):
        yield get_batch_eg(batch, idx=i)


class StoreResultCallback(Callback):
    _table: Optional[pa.Table] = None
    _writer: Optional[pq.ParquetWriter] = None

    def __init__(self, cache_dir: Optional[str] = None, store_fields: Optional[List[str]] = None):
        self.cache_dir = cache_dir
        if store_fields is None:
            store_fields = []
        self.store_fields = store_fields
        self._reset()

    def _reset(self):
        self.cache = tempfile.TemporaryFile(dir=self.cache_dir)
        self._close_writer()

    def on_predict_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: List[Any]
    ) -> None:
        self._close_writer()

    def on_predict_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Batch,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        batch = {k: v for k, v in batch.items() if k in self.store_fields}
        self._append_rows_cached_table({**batch, **outputs})

    def _close_writer(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def on_predict_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self._reset()

    def append_table_to_cache(self, table: pa.Table) -> None:
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.cache, table.schema)
        self._writer.write_table(table)

    def _append_rows_cached_table(self, batch: Batch) -> None:
        batch = cast_values_to_numpy(batch, as_contiguous=False)
        df = pd.DataFrame(iter_batch_rows(batch))
        # Convert from pandas to Arrow
        table = pa.Table.from_pandas(df)
        # write to the cache file
        self.append_table_to_cache(table)

    @property
    def data(self) -> pa.Table:
        return pq.read_table(self.cache)

    def iter_batches(self, batch_size=1000) -> Iterable[Batch]:
        for batch in self.data.to_batches(max_chunksize=batch_size):
            yield cast_values_to_numpy(batch.to_pydict())
