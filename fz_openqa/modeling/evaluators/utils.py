from typing import List

import torch

from fz_openqa.utils.datastruct import Batch


def check_first_doc_positive(batch):
    assert torch.all(batch["document.is_positive"][:, 0] == 1)
    if batch["document.is_positive"].shape[1] > 1:
        assert torch.all(batch["document.is_positive"][:, 1:] == 0)


def expand_and_flatten(batch: Batch, n_docs, *, keys: List[str]) -> Batch:
    output = {}
    for k in keys:
        v = batch[k]
        v = v[:, None].expand(v.shape[0], n_docs, *v.shape[1:]).contiguous()
        output[k] = v.view(-1, *v.shape[2:])
    return output


def flatten_first_dims(batch: Batch, n_dims, *, keys: List[str]) -> Batch:
    output = {}
    for k in keys:
        output[k] = batch[k].view(-1, *batch[k].shape[n_dims:])
    return output
