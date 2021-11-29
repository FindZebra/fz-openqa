import warnings
from copy import copy
from random import shuffle
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import rich
import torch
from datasets import Split
from omegaconf import DictConfig
from scipy.special import softmax
from torch import Tensor

from ...utils.pretty import pprint_batch
from ...utils.shape import infer_shape
from .base import Pipe
from .nesting import Nested
from .sorting import reindex
from fz_openqa.datamodules.pipes.control.condition import HasPrefix
from fz_openqa.utils.datastruct import Batch


class SelectDocs(Nested):
    "Select `total` documents with `max_pos_docs` positive documents (i.e. document.match_score>0)"

    def __init__(
        self,
        *,
        total: Union[int, Dict],
        max_pos_docs: Optional[int] = 1,
        pos_select_mode: str = "first",
        neg_select_mode: str = "first",
        strict: bool = False,
        shuffle: bool = False,
        prefix="document.",
        id="select-docs",
        **kwargs,
    ):
        pipe = SelectDocsOneEg(
            total=total,
            max_pos_docs=max_pos_docs or total,
            pos_select_mode=pos_select_mode,
            neg_select_mode=neg_select_mode,
            strict=strict,
            shuffle=shuffle,
        )

        super(SelectDocs, self).__init__(pipe=pipe, input_filter=HasPrefix(prefix), id=id, **kwargs)


class SelectDocsOneEg(Pipe):
    def __init__(
        self,
        *,
        total: Union[int, Dict],
        max_pos_docs: int = 1,
        pos_select_mode: str = "first",
        neg_select_mode: str = "first",
        strict: bool = True,
        shuffle: bool = False,
        score_key: str = "document.match_score",
        retrieval_score_key: str = "document.retrieval_score",
        **kwargs,
    ):
        super(SelectDocsOneEg, self).__init__(**kwargs)
        self.total = total
        self.score_key = score_key
        self.retrieval_score_key = retrieval_score_key
        self.max_pos_docs = max_pos_docs
        self.pos_select_mode = pos_select_mode
        self.neg_select_mode = neg_select_mode
        self.strict = strict
        self.shuffle = shuffle

    def output_keys(self, input_keys: List[str]) -> List[str]:
        return input_keys

    def _call_batch(self, batch: Batch, split: Optional[Split] = None, **kwargs) -> Batch:
        total = self.total
        if isinstance(total, (dict, DictConfig)):
            total = total[str(split)]

        # get the positive documents, if `score_key` is not available
        # consider all documents as negative
        if self.score_key in batch:
            is_positive = [x > 0 for x in batch[self.score_key]]
        else:
            warnings.warn(
                f"The key {self.score_key} was not found in batch with "
                f"keys={list(batch.keys())}. Handling all documents as negative."
            )
            values = next(iter(batch.values()))
            is_positive = len(values) * [False]
        assert len(is_positive) >= total

        # get the positive indexes
        # todo: check if prob select works as expected
        positive_idx = [i for i, x in enumerate(is_positive) if x]
        pos_probs = self.get_probs(batch, positive_idx)
        selected_positive_idx = select_values(
            positive_idx,
            k=min(self.max_pos_docs, len(positive_idx)),
            mode=self.pos_select_mode,
            probs=pos_probs,
        )

        # get the negative indexes
        negative_idx = [i for i, x in enumerate(is_positive) if not x]
        neg_probs = self.get_probs(batch, negative_idx)
        selected_negative_idx = select_values(
            negative_idx,
            k=total - len(selected_positive_idx),
            mode=self.neg_select_mode,
            probs=neg_probs,
        )

        # final index
        index = selected_positive_idx + selected_negative_idx

        index = self.check_index_consistency(
            index,
            negative_idx=negative_idx,
            positive_idx=positive_idx,
            selected_negative_idx=selected_negative_idx,
            total=total,
        )

        # shuffle
        if self.shuffle:
            shuffle(index)

        # re-index and return
        return {k: reindex(v, index) for k, v in batch.items()}

    def get_probs(self, batch, idx):
        scores = batch[self.retrieval_score_key][idx]
        if len(scores):
            if isinstance(scores, Tensor):
                probs = torch.softmax(scores.to(torch.float64), dim=-1)
            else:
                probs = softmax(scores)

            return probs
        else:
            return None

    def check_index_consistency(
        self,
        index,
        *,
        negative_idx,
        positive_idx,
        selected_negative_idx,
        total,
    ):
        if self.strict:
            # check output
            debug_str = (
                f"The resulting index is smaller (N={len(index)}) "
                f"than the expected size (total={total}). "
                f"You probably need to increase `max_pos_docs` or "
                f"reduce `total. "
            )
            assert len(index) == self.total, debug_str
        elif len(index) < total:
            if len(selected_negative_idx) == len(negative_idx):
                warnings.warn(
                    f"There were not enough negative documents to "
                    f"return the right number of selected "
                    f"documents (total={total}) with this "
                    f"value of max_pos_docs={self.max_pos_docs}. "
                    f"Completing the selection with more positive "
                    f"documents than specified with max_pos_docs. "
                    f"Use strict=True to raise an error instead. "
                    f"(n_positive={len(positive_idx)}, "
                    f"n_negative={len(negative_idx)})"
                )
                args = {
                    "k": total - len(negative_idx),
                    "mode": self.pos_select_mode,
                }
                # todo: select with probs
                selected_positive_idx = select_values(positive_idx, **args)
                index = selected_positive_idx + selected_negative_idx
            else:
                raise NotImplementedError
        return index


def select_values(
    values: List[int], *, k: int, mode: str = "first", probs: Optional[np.ndarray] = None
) -> List[int]:
    if mode == "first":
        return values[:k]
    elif mode == "sample":
        k = min(len(values), k)
        return [x for x in np.random.choice(copy(values), size=k, replace=False, p=probs)]
    else:
        raise ValueError(f"Unknown mode {mode}")
