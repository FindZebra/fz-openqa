from __future__ import annotations

import shutil
import tempfile
import unittest
import warnings
from pathlib import Path
from typing import Optional, List

import dill
import numpy as np
import torch
from parameterized import parameterized_class, parameterized

from fz_openqa.utils.datastruct import PathLike
from fz_openqa.utils.tensor_arrow import TensorArrowWriter, TORCH_DTYPES, TensorArrowTable

DSET_SIZE: int = 1000

warnings.filterwarnings("ignore")


@parameterized_class(('dtype_writer', 'dtype_reader', 'vec_shape'), [
    ('float32', 'float32', [10, ]),
    ('float16', 'float32', [10, ]),
    ('float32', 'float32', [10, 8, ]),
    ('float16', 'float32', [10, 8, ]),
    ('float32', 'float16', [10, ]),
    ('float16', 'float32', [10, ]),
    ('float32', 'float16', [10, 8, ]),
    ('float16', 'float32', [10, 8, ])
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
                tensor = torch.randn(self.writer_batch_size, *vec_shape)
                store.write(tensor)

                # store the raw vectors
                tensor = tensor.to(TORCH_DTYPES[dtype])
                if self.tensors is None:
                    self.tensors = tensor
                else:
                    self.tensors = torch.cat((self.tensors, tensor), dim=0)

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
        self.assertEqual(retrieved_tsr.shape, original_tsr.shape)
        self.assertTrue(torch.allclose(retrieved_tsr, original_tsr.to(retrieved_tsr)))
        self.assertEqual(retrieved_tsr.dtype, self.reader.dtype('torch'))

    def test_pickle(self):
        self.assertTrue(dill.pickles(self.reader))
