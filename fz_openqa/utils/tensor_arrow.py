from __future__ import annotations

import math
import multiprocessing as mp
import shutil
import tempfile
from copy import copy
from pathlib import Path
from typing import Any
from typing import BinaryIO
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pyarrow as pa
import rich
import torch
from datasets.features import numpy_to_pyarrow_listarray
from datasets.table import MemoryMappedTable
from loguru import logger

from fz_openqa.utils.datastruct import PathLike

TORCH_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "int64": torch.int64,
    "int32": torch.int32,
    "int8": torch.int8,
}

NUMPY_DTYPES = {
    "float16": "f2",
    "float32": "f4",
    "int64": "i8",
    "int32": "i4",
    "int8": "i1",
}

PA_DTYPES = {
    "float16": pa.float16(),
    "float32": pa.float32(),
    "int64": pa.int64(),
    "int32": pa.int32(),
    "int8": pa.int8(),
}

FORMAT_DTYPES = {"torch": TORCH_DTYPES, "numpy": NUMPY_DTYPES, "pyarrow": PA_DTYPES}

TENSOR_COL = "tensor"

ARRAY_INDEX = Union[int, slice, list, np.ndarray]


def get_dtype(format: str, dtype: str) -> str:
    return FORMAT_DTYPES[format][dtype]


def pad_dim_zero(v: torch.Tensor, max_length: int, pad_value: torch.Tensor):
    if len(v) >= max_length:
        return v[:max_length]
    else:
        filling = pad_value.expand(max_length - len(v), *v.shape[1:])
        return torch.cat([v, filling])


class ArrowWriters:
    """Handle multiple Arrow Writers"""

    _sinks: Optional[Dict[str, BinaryIO]] = None
    _writers: Optional[Dict[str, pa.RecordBatchFileWriter]] = None

    def __init__(self, paths: Dict[str, PathLike], schemas: Optional[Dict[str, pa.Schema]] = None):
        self.paths = paths
        self.schemas = schemas

    def open(self, schemas: Optional[Dict[str, pa.Schema]] = None):
        schemas = schemas or self.schemas
        assert all(k in self.paths for k in schemas.keys())
        self._sinks = {k: open(str(self.paths[k]), "wb") for k in schemas.keys()}
        assert len(self._sinks) == len(self.paths)
        self._writers = {
            k: pa.RecordBatchStreamWriter(self.paths[k], schema=schemas[k]) for k in schemas.keys()
        }

        return self

    def write(self, tables: Dict[str, pa.Table | pa.RecordBatch]):
        assert all(k in self.paths for k in tables.keys())
        assert self._writers is not None
        for k in tables.keys():
            self._writers[k].write(tables[k])

    def close(self):
        if self._writers is not None:
            for w in self._writers.values():
                w.close()
        if self._sinks is not None:
            for s in self._sinks.values():
                s.close()

        return self

    def __del__(self):
        self.close()


class TensorArrowBase:
    """Allow storing tensors as arrow objects
    for efficient memory-mapped loading"""

    def __init__(self, path: PathLike, dtype: str = "float32"):
        super().__init__()
        self.path = Path(path)
        self._dtype = dtype

    @property
    def vectors_path(self) -> Path:
        return self.path / "vectors.arrow"

    @property
    def index_path(self) -> Path:
        return self.path / "index.arrow"

    def _load_table(self, path: PathLike) -> MemoryMappedTable:
        return MemoryMappedTable.from_file(str(path))

    def dtype(self, format: str):
        return get_dtype(format, self._dtype)


class TensorArrowWriter(TensorArrowBase):
    _writer: Optional[ArrowWriters] = None
    _curr_vec_index: int
    _curr_row_index: int

    def __enter__(self):
        self.open()
        return self

    def _reset(self):
        self._curr_vec_index = 0
        self._curr_row_index = 0

    def open(self):
        self.path.mkdir(parents=True, exist_ok=True)
        paths = {
            "vectors": self.vectors_path,
            "index": self.index_path,
        }
        schemas = {
            "vectors": pa.schema([(TENSOR_COL, self.dtype("pyarrow"))]),
            "index": pa.schema(
                [("start", pa.int64()), ("stop", pa.int64()), ("shape", pa.list_(pa.int32(), -1))]
            ),
        }
        self._reset()
        self._writer = ArrowWriters(paths, schemas).open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        if exc_type is not None:
            shutil.rmtree(self.path)

    def close(self):
        self._writer.close()
        return self

    @torch.no_grad()
    def write(self, tensor: torch.Tensor, idx: Optional[List[int]] = None):
        if self._writer is None:
            raise RuntimeError("Writer not initialized")

        if idx is not None:
            if not is_contiguous_index(idx):
                raise ValueError(
                    "Index must be contiguous (i.e. x[i+1] = x[i] +1)." f"Received: {idx}"
                )

            if idx[0] != self._curr_row_index:
                raise ValueError(
                    f"Index must follows with stored indexes. "
                    f"idx[0]={idx[0]}, len(indexes)= {self._curr_row_index}"
                )

        # infer the shape of the tensor
        shape = tensor.shape[1:]
        batch_size = tensor.shape[0]
        stride = np.prod(shape)
        num_elements = tensor.numel()
        # flatten the tensor
        array = tensor.reshape(-1)

        # cast to pyarrow RecordBatch
        array = array.detach().cpu().numpy()
        array = array.astype(self.dtype("numpy"))
        array = pa.array(array, type=self.dtype("pyarrow"))
        vectors_batch = pa.record_batch([array], names=[TENSOR_COL])

        # Create the index RecordBatch
        start = np.arange(
            self._curr_vec_index, self._curr_vec_index + stride * batch_size, stride, dtype=np.int64
        )
        end = start + stride
        shape = np.array(shape, dtype=np.int16)
        shape = np.repeat(shape[None, :], batch_size, axis=0)
        shapes = numpy_to_pyarrow_listarray(shape, type=pa.int32())
        index_batch = pa.RecordBatch.from_arrays(
            [start, end, shapes], names=["start", "stop", "shape"]
        )

        self._writer.write({"vectors": vectors_batch, "index": index_batch})
        self._curr_vec_index += num_elements
        self._curr_row_index += batch_size


def is_contiguous_index(item: list | np.ndarray | torch.Tensor) -> bool:
    """check if x[i+1] = x[i] + 1"""
    if isinstance(item, list):
        return [i + 1 for i in item[:-1]] == item[1:]
    elif isinstance(item, (np.ndarray, torch.Tensor)):
        return (item[:-1] + 1 == item[1:]).all()
    else:
        raise TypeError(f"{type(item)} is not supported")


class TensorArrowTable(TensorArrowBase):
    """
    An Arrow table optimized for Tensors. This allows for:
    1. zero-copy reads of arbitrary-shaped tensors
    2. Fast random access using HuggingFace's `MappedMemoryTable`

    Notes
    -----
    Limitations:
    1. The table can only handle one column of tensors. But this should be easy to extend.
    2. The table can only handle tensors with the same shape, or with varying dim zero.

    Future work:
    -----------
    1. todo: Cleanup the behaviour for non-equal shapes
    2. todo: Extend non-equal shapes handling to other dims than zero
    """

    _index: Optional[torch.Tensor] = None
    _shapes: Optional[torch.Tensor] = None
    _shared_shape: Optional[torch.Tensor] = None
    _vectors: Optional[MemoryMappedTable] = None
    _no_pickle_list: List[str] = ["_index", "_shapes", "_vectors"]
    _equal_shapes: bool = False

    def __init__(self, *args, pad_value: Any = torch.nan, **kwargs):
        super().__init__(*args, **kwargs)
        self._load_all()
        self.pad_value = pad_value

    def _load_all(self):
        self._build_index()
        self._vectors = self._load_table(self.vectors_path)

    @staticmethod
    def are_shapes_equal(shapes: torch.Tensor):
        return torch.all(shapes == shapes[:1])

    @torch.no_grad()
    def _build_index(self):
        index_table = self._load_table(self.index_path)
        columns = index_table.column_names
        start = index_table.column(columns.index("start"))
        stop = index_table.column(columns.index("stop"))
        start = torch.from_numpy(start.to_numpy())
        stop = torch.from_numpy(stop.to_numpy())
        self._index = torch.cat([start[:, None], stop[:, None]], dim=1).contiguous()

        shapes = index_table.column(columns.index("shape"))
        self._shapes = torch.tensor(shapes.to_pylist())

        if self.are_shapes_equal(self._shapes):
            self._equal_shapes = True
            self._shared_shape = self._shapes[0]
            logger.debug(
                f"All shapes are equal. Using a the reference shape: {self._shared_shape}."
            )

        else:
            n_unique_per_dims = [
                len(torch.unique(self._shapes[:, i])) for i in range(self._shapes.shape[1])
            ]
            dim_unequal_numel = [i for i, n in enumerate(n_unique_per_dims) if n != 1]
            if len(dim_unequal_numel) > 1:
                raise NotImplementedError("Cannot handle with more than one varying dimensions")
            dim_unequal_numel = dim_unequal_numel[0]
            if dim_unequal_numel != 0:
                raise NotImplementedError(
                    f"Cannot handle varying dimension other " f"than dim 0, dim={dim_unequal_numel}"
                )

            # set the shapes as [-1, *dims]
            self._shared_shape = self._shapes[0].clone()
            self._shared_shape[0] = -1
            logger.debug(
                f"Not all shapes are equal. Using the reference shape: {self._shared_shape}."
            )

    @property
    def vectors(self) -> MemoryMappedTable:
        if self._vectors is None:
            self._vectors = self._load_table(self.vectors_path)
        return self._vectors

    @property
    def num_rows(self) -> int:
        return self._load_table(self.index_path).num_rows

    def __len__(self):
        return len(self._index)

    @property
    def nbytes(self) -> int:
        return self.vectors.nbytes + self._load_table(self.index_path).nbytes

    def __call__(self, index: ARRAY_INDEX) -> torch.Tensor:
        return self.__getitem__(index)

    @property
    def shape(self) -> Tuple[int, ...]:
        return (len(self), *(int(x) for x in self._shared_shape))

    def size(self, dim: int = None):
        if dim is None:
            return self.shape
        else:
            return self.shape[dim]

    def dim(self) -> int:
        return len(self.shape)

    def view(self, *new_shape: int) -> "TensorArrowTable":
        """Hacky implementation used to flatten tensors for the ColbertIndex."""
        if len(new_shape) != 2 or (new_shape[0] != -1 or new_shape[1] == -1):
            raise NotImplementedError(
                "Only implemented for flattening. " "Accepted shapes are of the form (-1, x)"
            )
        if not self._equal_shapes:
            raise NotImplementedError("Only implemented for constant shapes")

        # create a new table with the new shape
        new_table = copy(self)

        dim = new_shape[1]
        n_elements = new_table._index[-1, -1] - new_table._index[0, 0]
        if n_elements % dim != 0:
            raise ValueError(
                "The number of elements in the new shape must be a "
                "multiple of the number of elements in the old shape. "
                f"Found total={n_elements} and h={dim}, "
                f"total % h = {n_elements % dim}."
            )

        # build the new index
        new_index = torch.arange(0, n_elements + 1, dim, dtype=torch.int64)
        rich.print(f"> n: {n_elements + 1}")
        rich.print(f"> new_index.start: {new_index[:10]}")
        rich.print(f"> new_index.end: {new_index[-10:]}")
        new_table._index = torch.stack([new_index[:-1], new_index[1:]], dim=1)
        if not new_table._index[0, 0] == self._index[0, 0]:
            raise ValueError(
                f"The first index in the new view must be the same "
                f"as the first index in the old shape. "
                f"Found {new_table._index[0, 0]} "
                f"and {self._index[0, 0]}."
            )
        if not new_table._index[-1, -1] == self._index[-1, -1]:
            raise ValueError(
                f"The last index in the new view must be the same "
                f"as the last index in the old shape. "
                f"Found {new_table._index[-1, -1]} "
                f"and {self._index[-1, -1]}."
            )

        # set the shapes
        new_table._shared_shape = torch.tensor(
            [
                dim,
            ],
            dtype=torch.int64,
        )
        new_table._shapes = new_table._shared_shape.repeat(len(new_table), 1)
        return new_table

    @torch.no_grad()
    def __getitem__(self, item: ARRAY_INDEX) -> Optional[torch.Tensor]:
        is_int = False
        is_slice = False
        if isinstance(item, int):
            if item < 0:
                item += self.num_rows
            item = slice(item, item + 1)
            is_int = True
        elif isinstance(item, slice):
            is_slice = True
        elif isinstance(item, (list, np.ndarray)):
            if len(item) == 0:
                return None
            if is_contiguous_index(item):
                item = slice(item[0], item[-1] + 1)
                is_slice = True
        else:
            raise ValueError(f"Invalid item type {type(item)}")

        index = self._index[item]
        batch_size = len(index)

        # get the shape
        shape = self._shared_shape.clone()
        vec_shapes = self._shapes[item].clone()

        if is_slice:
            # if vectors are contiguous, query the table using a single call
            start = index[0, 0]
            stop = index[-1, -1]
            vectors = self.vectors.fast_slice(offset=start, length=stop - start)

            vectors = vectors[TENSOR_COL]
            vectors = vectors.to_numpy()
            vectors = torch.from_numpy(vectors)

            if not self._equal_shapes:
                # if the vectors are not of the same shape, split into separate vectors
                vectors = list(self._extract_non_equally_shaped_vectors(vectors, shapes=vec_shapes))
        else:
            vectors = []
            for start, stop in index:
                # query the table using a call for each vector
                vec = self._vectors.fast_slice(start, length=stop - start)
                vec = vec[TENSOR_COL]
                vec = vec.to_numpy()
                vec = torch.from_numpy(vec)
                vectors.append(vec)

        if isinstance(vectors, list):
            if not self._equal_shapes:
                # potentially pad the vectors to max shape
                vectors = self._pad_vectors_to_max_length(vectors, shape=self._shared_shape)
                shape[0] = len(vectors[0])

            # concatenate the vectors
            vectors = torch.cat([v[None] for v in vectors], dim=0)

        # reshape the vectors and cast
        vectors = vectors.view(batch_size, *shape)
        if vectors.dtype != self.dtype("torch"):
            logger.warning(
                f"Converting vectors from {vectors.dtype} to {self.dtype('torch')}. "
                f"Make sure to save the vectors with the same dtype."
            )
            vectors = vectors.to(self.dtype("torch"))

        if is_int:
            vectors = vectors[0]
        return vectors

    def _extract_non_equally_shaped_vectors(self, vector, shapes):
        i = 0
        for shape in shapes:
            numel = shape.prod()

            vec = vector[i : i + numel]
            i += numel
            yield vec

    def _pad_vectors_to_max_length(self, vectors, shape):
        vectors = [v.view(*shape) for v in vectors]
        max_length = max([v.shape[0] for v in vectors])
        pad_value = vectors[0][:1].clone()
        pad_value.fill_(self.pad_value)
        vectors = [pad_dim_zero(v, max_length, pad_value) for v in vectors]
        return vectors

    def to_batches(self, max_chunksize: int = 1000) -> Iterator[torch.Tensor]:
        for i in range(0, self.num_rows, max_chunksize):
            yield self[i : i + max_chunksize]

    def parallel_fetch(
        self, indexes: Iterable[ARRAY_INDEX], num_workers: int = 4
    ) -> Iterable[torch.Tensor]:
        pool = mp.Pool(num_workers)
        return pool.imap(self.__getitem__, indexes)

    def __getstate__(self):
        state = self.__dict__.copy()
        for k in self._no_pickle_list:
            state.pop(k, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        for k in self._no_pickle_list:
            self.__dict__[k] = None
        self._load_all()

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(shape={self.shape}, "
            f"dtype={self._dtype}, pad_value={self.pad_value})"
        )


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmpdir:
        bs = 100
        path = tmpdir
        dtype = "float16"
        length = 20
        size = 1000

        # data
        data = torch.randn(size, length, 32, dtype=torch.float16)
        flat_data = data.view(-1, 32)

        # store
        with TensorArrowWriter(path, dtype=dtype) as writer:
            for i in range(0, data.size(0), bs):
                L = None
                rows = data[i : i + bs, :L]
                writer.write(rows)

        # load
        vectors = TensorArrowTable(path, dtype=dtype)
        rich.print(vectors)
        rich.print(vectors[[1, 4, 34, 99]].sum(-1))

        # view
        flat_vectors = vectors.view(-1, vectors.size(-1))
        if flat_vectors.shape != flat_data.shape:
            raise Exception(
                "view shape mismatch. "
                f"Found {flat_vectors.shape}, "
                f"expected {flat_data.shape}"
            )

        for idx in range(100):
            i = np.random.randint(0, len(flat_data))
            if not (flat_data[i] == flat_vectors[i]).all():
                rich.print(f"Computed: {flat_data[i]}")
                rich.print(f"Reference: {flat_vectors[i]}")
                raise Exception(f"{i} mismatch")
