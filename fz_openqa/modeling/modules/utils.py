from typing import List

import torch

from fz_openqa.utils.datastruct import Batch


def check_only_first_doc_positive(batch, *, match_key="document.match_score"):
    assert torch.all(
        batch[match_key][:, 0] > 0
    ), "Not all documents with index 0 are positive."
    if batch[match_key].shape[1] > 1:
        assert torch.all(
            batch[match_key][:, 1:] == 0
        ), "Not all documents with index >0 are negative."


def expand_and_flatten(batch: Batch, n_docs, *, keys: List[str]) -> Batch:
    output = {}
    for k in keys:
        v = batch[k]
        v = v[:, None].expand(v.shape[0], n_docs, *v.shape[1:]).contiguous()
        output[k] = v.view(-1, *v.shape[2:])
    return output


def flatten_first_dims(batch: Batch, n_dims, *, keys: List[str]) -> Batch:
    for k in keys:
        batch[k] = batch[k].view(-1, *batch[k].shape[n_dims:])
    return batch
