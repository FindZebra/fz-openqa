import os
import random  # type: ignore
import shutil
import time
import timeit
import warnings
from pathlib import Path

import numpy as np
import psutil
import rich
import torch
from tqdm.rich import tqdm

from fz_openqa.utils.io import display_file_size
from fz_openqa.utils.tensor_arrow import TensorArrowTable
from fz_openqa.utils.tensor_arrow import TensorArrowWriter
from fz_openqa.utils.tensor_arrow import TORCH_DTYPES

warnings.filterwarnings("ignore")


DSET_SIZE = 100000
BATCH_SIZE = 1000
VEC_SHAPE = (200, 32)

if __name__ == "__main__":

    path = Path("cache/tensors/")
    dtype = "float32"
    with TensorArrowWriter(path, dtype=dtype) as store:
        for i in tqdm(range(0, DSET_SIZE, BATCH_SIZE), desc="Writing"):
            tensor = torch.randn(BATCH_SIZE, *VEC_SHAPE, dtype=TORCH_DTYPES[dtype])
            store.write(tensor)

    # get the size of the original tensor and delete it
    one_tensor_nbytes = tensor[0].element_size() * tensor[0].nelement()
    dataset_nbytes = DSET_SIZE * one_tensor_nbytes
    del tensor

    # file disk size
    for f in path.iterdir():
        display_file_size(str(f), f, print_fn=rich.print)

    # read all tensors
    reader = TensorArrowTable(path, dtype=dtype)
    rich.print(f"Written num_rows={reader.num_rows}, bytes={dataset_nbytes >> 20} MB")
    index = store._load_table(reader.index_path)
    vectors = store._load_table(store.vectors_path)
    rich.print(f"Number of vector values: {vectors.num_rows}")
    del reader
    time.sleep(0.5)

    # memory profiling
    mem_before = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    reader = TensorArrowTable(path, dtype=dtype)
    mem_after = psutil.Process(os.getpid()).memory_info().rss >> 20
    rich.print(f"RAM memory used: {(mem_after - mem_before):.3f} MB")

    # loading speed profiling (sweep)
    batch_size = 1000

    start = time.time()
    for i in range(0, len(reader), batch_size):
        vectors = reader[i : i + batch_size]
    elapsed_time = time.time() - start

    rich.print(
        f"Time to sweep over the {dataset_nbytes / 1024 ** 3:.3f} GB "
        f"dataset: {elapsed_time:.3f} sec, "
        f"ie. {float(dataset_nbytes / 1024 ** 3) / elapsed_time:.3f} GB/s, "
        f"{float(len(reader)) / elapsed_time:.1e} rows/s"
    )

    # loading speed profiling (random access)
    batch_size = 1000
    n_samples = min(10000, len(reader))
    index = list(range(n_samples))
    np.random.shuffle(index)

    start = time.time()
    for i in range(0, n_samples, batch_size):
        items = index[i : i + batch_size]
        vectors = reader[items]

    elapsed_time = time.time() - start
    rich.print(
        f"Time to fetch the {n_samples * one_tensor_nbytes / 1024 ** 2:.3f} MB "
        f"samples: {elapsed_time:.3f} sec, "
        f"ie. {float(n_samples * one_tensor_nbytes / 1024 ** 2) / elapsed_time:.3f} MB/s, "
        f"{float(n_samples) / elapsed_time:.1e} rows/s"
    )

    # loading speed profiling (random access, multiprocessing)
    indexes = [index[i : i + batch_size] for i in range(0, n_samples, batch_size)]
    loader = reader.parallel_fetch(indexes, num_workers=4)

    start = time.time()
    for vec in loader:
        vec.sum()
    elapsed_time = time.time() - start

    rich.print(
        f"Parallel: Time to fetch the {n_samples * one_tensor_nbytes / 1024 ** 2:.3f} MB "
        f"samples: {elapsed_time:.3f} sec, "
        f"ie. {float(n_samples * one_tensor_nbytes / 1024 ** 2) / elapsed_time:.3f} MB/s, "
        f"{float(n_samples) / elapsed_time:.1e} rows/s"
    )

    shutil.rmtree(path)
