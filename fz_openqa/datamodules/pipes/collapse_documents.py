from typing import List
from typing import Optional

import torch
from torch import Tensor

from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.utils.datastruct import Batch


def pad(x: Tensor, length: int, fill_value=0):
    """
    Pad a tensor to a given length.
    """
    if x.size(0) == length:
        return x
    return torch.cat([x, x.new_full((length - x.size(0),) + x.size()[1:], fill_value)])


def unique(ids):
    """ "
    Return the unique elements, the inverse index and the index to
    go from the original ids to the unique_ids

    Inspired by https://github.com/pytorch/pytorch/issues/36748"""
    unique_ids, inv_ids = torch.unique(ids, return_inverse=True)
    map_idx = torch.arange(ids.shape[0])
    map_idx = unique_ids.new_empty(unique_ids.size(0)).scatter_(0, inv_ids, map_idx)
    return unique_ids, inv_ids, map_idx


class SqueezeDocuments(Pipe):
    def __init__(
        self,
        *,
        field: str = "document",
        id_key: str = "row_idx",
        keys: List[str] = None,
        score_key: str = "retrieval_score",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if keys is None:
            keys = ["input_ids", "attention_mask"]
        self.field = field
        self.id_key = id_key
        self.score_key = score_key
        self.keys = keys

    def _call_batch(self, batch: Batch, idx: Optional[List[int]] = None, **kwargs) -> Batch:
        output = {}
        for key in self.keys:
            x = batch[f"{self.field}.{key}"]
            x = x[:, 0]
            output[f"{self.field}.{key}"] = x

        return output


class CollapseDocuments(Pipe):
    def __init__(
        self,
        *,
        field: str = "document",
        id_key: str = "row_idx",
        keys: List[str] = None,
        score_key: str = "retrieval_score",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if keys is None:
            keys = ["input_ids", "attention_mask"]
        self.field = field
        self.id_key = id_key
        self.score_key = score_key
        self.keys = keys

    def _call_batch(self, batch: Batch, idx: Optional[List[int]] = None, **kwargs) -> Batch:
        ids = batch[f"{self.field}.{self.id_key}"]
        original_shape = ids.shape
        unique_ids, inv_ids, map_idx = [], [], []

        # get unique ids for each batch position
        for i in range(original_shape[0]):
            unique_ids_i, inv_ids_i, map_idx_i = unique(ids[i].view(-1))

            unique_ids.append(unique_ids_i)
            inv_ids.append(inv_ids_i)
            map_idx.append(map_idx_i)

        max_length = max(len(x) for x in map_idx)

        # concatenate all ids
        # unique_ids = torch.cat([pad(x, max_length) for x in unique_ids], dim=0)
        inv_ids = torch.cat([x[None] for x in inv_ids], dim=0)
        map_idx = torch.cat([pad(x, max_length)[None] for x in map_idx], dim=0)

        output = {f"{self.field}.inv_idx": inv_ids}
        for key in self.keys:
            x = batch[f"{self.field}.{key}"]
            x = x.view(original_shape[0], -1, *x.shape[len(original_shape) :])
            map_idx_ = map_idx
            for _ in range(max(0, len(x.shape) - len(map_idx.shape))):
                map_idx_ = map_idx_.unsqueeze(-1)
            map_idx_ = map_idx_.expand(*map_idx_.shape[:2], *x.shape[2:])
            unique_x = x.gather(dim=1, index=map_idx_)
            output[f"{self.field}.{key}"] = unique_x

        return output
