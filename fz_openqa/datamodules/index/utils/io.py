from __future__ import annotations

import time
from logging import Logger
from typing import Optional
from typing import Tuple

import numpy as np
import rich
import torch
from torch import Tensor

from fz_openqa.utils.tensor_arrow import TensorArrowTable


def read_vectors_from_table(
    vectors_table: Tensor | TensorArrowTable,
    boundaries: Optional[Tuple[int, int]] = None,
    logger: Optional[Logger] = None,
) -> Tensor:
    """
    Read vectors from a `TensorArrowTable` given `boundarieas`.
    Parameters
    ----------
    vectors_table:
        A `TensorArrowTable` or a `Tensor`
    boundaries
        A tuple of (start, end) indices (start: inclusive, end: exclusive)
    verbose
        If True, log the shape and memory size of the returned tensor.

    Returns
    -------
    torch.Tensor
    the Tensor of vectors corresponding to the boundaries

    """

    if boundaries is None:
        boundaries = (0, len(vectors_table))
    assert len(boundaries) == 2

    if logger is not None and isinstance(vectors_table, TensorArrowTable):
        mem_size = vectors_table.nbytes // 1024 ** 3
        logger.info(
            f"Reading {boundaries[1] - boundaries[0]} vectors({mem_size:.3f} GB), "
            f"boundaries={boundaries}, "
            f"from {vectors_table.path}"
        )
    start_time = time.time()

    # reade the vectors
    vectors = vectors_table[boundaries[0] : boundaries[1]]

    if logger is not None and isinstance(vectors_table, TensorArrowTable):
        logger.info(
            f"Read vectors of shape {vectors.shape}, dtype={vectors.dtype}. "
            f"Elapsed time: {time.time() - start_time:.3f}s"
        )
    return vectors


def log_mem_size(x, msg, logger=None):
    if isinstance(x, torch.Tensor):
        mem_size = x.element_size() * x.nelement()
    elif isinstance(x, np.ndarray):
        mem_size = x.dtype.itemsize * x.size
    else:
        raise TypeError(f"Unsupported type {type(x)}")
    mem_size /= 1024 ** 2
    prec = "MB"
    if mem_size > 1024:
        mem_size /= 1024
        prec = "GB"
    msg = f"{msg} mem. size={mem_size:.3f} {prec}, shape={x.shape}, dtype={x.dtype}"
    if logger is None:
        rich.print(msg)
    else:
        logger.info(msg)


def build_emb2pid_from_vectors(vectors_table: Tensor | TensorArrowTable) -> torch.Tensor:
    """WARNING: assumes all vectors are of the same length"""
    emb2pid = torch.arange(len(vectors_table), dtype=torch.long)
    one_vec = vectors_table[0]
    vector_length = one_vec.shape[0]
    emb2pid = emb2pid[:, None].expand(-1, vector_length)
    return emb2pid.reshape(-1).contiguous()
