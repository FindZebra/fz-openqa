from __future__ import annotations

import shutil
import tempfile
import unittest
import warnings
from copy import copy
from pathlib import Path
from typing import Optional, List

import dill
import numpy as np
import rich
import torch
from parameterized import parameterized_class, parameterized

from fz_openqa.utils.datastruct import PathLike
from fz_openqa.utils.tensor_arrow import TensorArrowWriter, TORCH_DTYPES, TensorArrowTable

DSET_SIZE: int = 1000
MAX_LENGTH:int = 20

warnings.filterwarnings("ignore")


def pad_dim_one(v: torch.Tensor, max_length: int, pad_value: torch.Tensor):
    if v.shape[1] >= max_length:
        return v[:, :max_length]
    else:
        p = pad_value.expand(-1, max_length - v.shape[1], *v.shape[2:])
        return torch.cat([v, p], dim=1)


@parameterized_class(('dtype_writer', 'dtype_reader', 'varying_dim', 'vec_shape'), [
    ('float32', 'float32', None, [10, ]),
    ('float16', 'float32', None, [10, ]),
    ('float32', 'float32', None, [10, 8, ]),
    ('float16', 'float32', None, [10, 8, ]),
    ('float32', 'float16', None, [10, ]),
    ('float16', 'float32', None, [10, ]),
    ('float32', 'float16', None, [10, 8, ]),
    ('float16', 'float32', None, [10, 8, ]),
    ('float32', 'float32', 0, [10, ]),
    ('float16', 'float32', 0, [10, ]),
    ('float32', 'float32', 0, [10, 8, ]),
    ('float16', 'float32', 0, [10, 8, ]),
    ('float32', 'float16', 0, [10, ]),
    ('float16', 'float32', 0, [10, ]),
    ('float32', 'float16', 0, [10, 8, ]),
    ('float16', 'float32', 0, [10, 8, ])
])
class TestTensorArrow(unittest.TestCase):
    writer_batch_size: int = 100

    tensors: Optional[torch.Tensor] = None

    def setUp(self) -> None:
        self.root = Path(tempfile.mktemp())

        # write vectors
        self._write(self.root, self.dtype_writer, self.vec_shape)
        # setup reader
        self.reader = TensorArrowTable(self.root, dtype=self.dtype_reader)

    def _write(self, path: PathLike, dtype: str, vec_shape: List[int]):
        with TensorArrowWriter(path, dtype=dtype) as store:
            for i in range(0, DSET_SIZE, self.writer_batch_size):
                batch_shape = copy(vec_shape)
                if self.varying_dim is not None:
                    batch_shape[self.varying_dim] = np.random.randint(10, MAX_LENGTH)
                tensor = torch.randn(self.writer_batch_size, *batch_shape)
                store.write(tensor)

                # store the raw vectors
                tensor = tensor.to(TORCH_DTYPES[dtype])

                # pad to max length
                if self.varying_dim is not None:
                    pad_value = tensor[:, :1]
                    pad_value.fill_(torch.nan)
                    tensor = pad_dim_one(tensor, MAX_LENGTH, pad_value)

                if self.tensors is None:
                    self.tensors = tensor
                    self.lengths = [batch_shape[0] for _ in range(self.writer_batch_size)]
                else:
                    self.tensors = torch.cat((self.tensors, tensor), dim=0)
                    self.lengths += [batch_shape[0] for _ in range(self.writer_batch_size)]

    def tearDown(self) -> None:
        shutil.rmtree(self.root)

    @parameterized.expand([
        (7,),
        (-1,),
        (slice(None, 3),),
        ([0, 1, 3],),
        (np.array([0, 1, 3]),),
        (slice(10, 13),),
        ([10, 11, 12],),
        (np.array([10, 11, 12]),),
        (np.random.randint(0, DSET_SIZE, size=10),),
    ])
    def test_indexing(self, indices: int | slice | np.ndarray):
        retrieved_tsr = self.reader[indices]
        original_tsr = self.tensors[indices]

        if isinstance(indices, int):
            retrieved_tsr = retrieved_tsr[None]
            original_tsr = original_tsr[None]

        # pad
        if self.varying_dim is not None:
            pad_value = retrieved_tsr[:, :1]
            pad_value.fill_(torch.nan)
            retrieved_tsr = pad_dim_one(retrieved_tsr, MAX_LENGTH, pad_value)

        # fill padding
        retrieved_tsr[retrieved_tsr!=retrieved_tsr] = 0
        original_tsr[original_tsr != original_tsr] = 0

        self.assertEqual(retrieved_tsr.shape, original_tsr.shape)
        self.assertTrue(torch.allclose(retrieved_tsr, original_tsr.to(retrieved_tsr)))
        self.assertEqual(retrieved_tsr.dtype, self.reader.dtype('torch'))

    def test_pickle(self):
        self.assertTrue(dill.pickles(self.reader))
