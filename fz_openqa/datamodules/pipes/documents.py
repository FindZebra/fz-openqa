from random import choices
from typing import List

import rich

from .base import Pipe
from .base import PrintBatch
from .nesting import Nested
from .sorting import reindex
from fz_openqa.utils.datastruct import Batch


class SelectDocs(Nested):
    def __init__(
        self,
        *,
        total: int,
        max_pos_docs: int = 1,
        pos_select_mode: str = "sample",
        neg_select_mode: str = "first",
        prefix="document.",
    ):
        pipe = SelectDocsEg(
            total=total,
            max_pos_docs=max_pos_docs,
            pos_select_mode=pos_select_mode,
            neg_select_mode=neg_select_mode,
        )
        super(SelectDocs, self).__init__(
            pipe=pipe, filter=lambda key: str(key).startswith(prefix)
        )


class SelectDocsEg(Pipe):
    def __init__(
        self,
        *,
        total: int,
        max_pos_docs: int = 1,
        pos_select_mode: str = "sample",
        neg_select_mode: str = "first",
    ):
        self.total = total
        self.max_pos_docs = max_pos_docs
        self.pos_select_mode = pos_select_mode
        self.neg_select_mode = neg_select_mode

    def __call__(self, batch: Batch, **kwargs) -> Batch:
        is_positive = batch["document.is_positive"]
        assert len(is_positive) >= self.total

        # get the positive indexes
        positive_idx = [i for i, x in enumerate(is_positive) if x]
        positive_idx = select_values(
            positive_idx, k=self.max_pos_docs, mode=self.pos_select_mode
        )

        # get the negative indexes
        negative_idx = [i for i, x in enumerate(is_positive) if not x]
        negative_idx = select_values(
            negative_idx,
            k=self.total - len(positive_idx),
            mode=self.neg_select_mode,
        )

        # final index
        index = positive_idx + negative_idx

        return {k: reindex(v, index) for k, v in batch.items()}


def select_values(
    values: List[int], *, k: int, mode: str = "first"
) -> List[int]:

    if mode == "first":
        return values[:k]
    elif mode == "sample":
        k = min(len(values), k)
        return choices(values, k=k)
    else:
        raise ValueError(f"Unknown mode {mode}")
