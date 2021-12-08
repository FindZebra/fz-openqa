import logging
import uuid
from pathlib import Path
from typing import Iterable
from typing import List
from typing import Optional
from typing import Union

import pyarrow as pa
import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback

from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.datastruct import PathLike
from fz_openqa.utils.tensor_arrow import TensorArrowTable
from fz_openqa.utils.tensor_arrow import TensorArrowWriter

logger = logging.getLogger(__name__)

IDX_COL = "__idx__"


def select_field_from_output(batch: Batch, fields: List[str]) -> torch.Tensor:
    avail_fields = [k for k in batch.keys() if k in fields]
    if len(avail_fields) != 1:
        raise ValueError(
            f"One and only one field in {fields} must be provided, " f"founds keys={batch.keys()})."
        )
    vector = [v for k, v in batch.items() if k in fields][0]
    return vector


class StorePredictionsCallback(Callback):
    """Allows storing the output of each `prediction_step` into a `pyarrow` table.
    The Table can be access using the attribute `table` or using the method `iter_batches()`"""

    _table: Optional[TensorArrowTable] = None
    _writer: Optional[TensorArrowWriter] = None
    _sink: Optional[pa.OSFile] = None
    cache_file: Optional[Union[str, Path]] = None

    def __init__(
        self,
        accepted_fields: Optional[List[str]],
        dtype: str = "float32",
        cache_file: Optional[PathLike] = None,
    ):
        if cache_file is None:
            cache_file = str(uuid.uuid4().hex)

        self.cache_file = Path(cache_file)
        self.accepted_fields = accepted_fields
        self.dtype = dtype
        self._reset()
        logger.info(f"Initialized {type(self).__name__} with cache_file={self.cache_file}")
        logger.info(f"is_written={self.is_written}")

    def on_predict_epoch_end(self, *args, **kwargs) -> None:
        self.close_writer()

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
        vector = select_field_from_output(outputs, self.accepted_fields)
        self._write_batch(vector, idx=batch.get(IDX_COL, None))

    def _reset(self):
        """get a new temporary cache file and close any existing writer."""
        self.close_writer()

    @property
    def is_written(self):
        return self.cache_file.exists()

    def close_writer(self):
        """close any existing writer"""
        if self._writer is not None:
            self._writer.close()

    def _write_batch(self, vectors: torch.Tensor, idx: Optional[List[int]] = None) -> None:
        """write a table to file."""
        if self._writer is None:
            self._writer = TensorArrowWriter(self.cache_file, dtype=self.dtype)
            self._writer.open()

        self._writer.write(vectors, idx=idx)

    @property
    def table(self) -> TensorArrowTable:
        """Returns the cached `pyarrow.Table`"""
        if self._table is None:
            self._table = TensorArrowTable(self.cache_file)
        return self._table

    def iter_batches(self, *args, **kwargs) -> Iterable[torch.Tensor]:
        """Returns an iterator over the cached batches"""
        yield from self.table.to_batches(*args, **kwargs)

    def __del__(self):
        self.close_writer()
        if self._table is not None:
            del self._table
