import warnings
from copy import copy
from typing import List
from typing import Optional

import numpy as np
import rich

from .base import Pipe
from .nesting import Nested
from .sorting import reindex
from fz_openqa.utils.datastruct import Batch


ARE_DOCS_SELECTED_KEY = "__doc_selected__"


class SelectDocs(Nested):
    def __init__(
        self,
        *,
        total: int,
        max_pos_docs: Optional[int] = 1,
        pos_select_mode: str = "sample",
        neg_select_mode: str = "first",
        strict: bool = False,
        prefix="document.",
    ):
        pipe = SelectDocsEg(
            total=total,
            max_pos_docs=max_pos_docs or total,
            pos_select_mode=pos_select_mode,
            neg_select_mode=neg_select_mode,
            strict=strict,
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
        strict: bool = True,
    ):
        self.total = total
        self.max_pos_docs = max_pos_docs
        self.pos_select_mode = pos_select_mode
        self.neg_select_mode = neg_select_mode
        self.strict = strict

    def __call__(self, batch: Batch, **kwargs) -> Batch:
        match_score = batch["document.match_score"]
        assert len(match_score) >= self.total

        # get the positive indexes
        positive_idx = [i for i, x in enumerate(match_score) if x]
        selected_positive_idx = select_values(
            positive_idx,
            k=min(self.max_pos_docs, len(positive_idx)),
            mode=self.pos_select_mode,
        )

        # get the negative indexes
        negative_idx = [i for i, x in enumerate(match_score) if not x]
        selected_negative_idx = select_values(
            negative_idx,
            k=self.total - len(selected_positive_idx),
            mode=self.neg_select_mode,
        )

        # final index
        index = selected_positive_idx + selected_negative_idx

        if self.strict:
            # check output
            debug_str = (
                f"The resulting index is smaller (N={len(index)}) "
                f"than the expected size (total={self.total}). "
                f"You probably need to increase `max_pos_docs` or "
                f"reduce `total. "
            )
            assert len(index) == self.total, debug_str
        elif len(index) < self.total:
            if len(selected_negative_idx) == len(negative_idx):
                warnings.warn(
                    f"There were not enough negative documents to "
                    f"return the right number of selected "
                    f"documents (total={self.total}) with this "
                    f"value of max_pos_docs={self.max_pos_docs}. "
                    f"Completing the selection with more positive "
                    f"documents than specified with max_pos_docs. "
                    f"Use strict=True to raise an error instead. "
                    f"(n_positive={len(positive_idx)}, "
                    f"n_negative={len(negative_idx)})"
                )
                args = {
                    "k": self.total - len(negative_idx),
                    "mode": self.pos_select_mode,
                }
                selected_positive_idx = select_values(positive_idx, **args)
                index = selected_positive_idx + selected_negative_idx
            else:
                rich.print(
                    f"=== SelectDocs: debugging ===\n"
                    f"len(index)({len(index)})<total({self.total}). \n"
                    f"n_selected_positive={len(selected_positive_idx)}, \n"
                    f"n_selected_negative={len(selected_negative_idx)}, \n"
                    f"n_positive={len(positive_idx)}, \n"
                    f"n_negative={len(negative_idx)}"
                )
                raise NotImplementedError

                # re-index and return
        return {
            ARE_DOCS_SELECTED_KEY: True,
            **{k: reindex(v, index) for k, v in batch.items()},
        }


def select_values(
    values: List[int], *, k: int, mode: str = "first"
) -> List[int]:
    if mode == "first":
        return values[:k]
    elif mode == "sample":
        k = min(len(values), k)
        return [
            x for x in np.random.choice(copy(values), size=k, replace=False)
        ]
    else:
        raise ValueError(f"Unknown mode {mode}")
