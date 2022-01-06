from copy import deepcopy
from typing import List

import dill
import pyarrow as pa
from datasets.features import numpy_to_pyarrow_listarray

from fz_openqa.datamodules.index.search_result import SearchResult
from fz_openqa.utils.datastruct import Batch


class FakeIndex:
    """A small class to test Search corpus without using a proper index"""

    index_name = "<name>"

    def search(self, *, query: Batch, k: int, **kwargs) -> SearchResult:
        values = query["question.text"]
        return SearchResult(
            index=[[0 for _ in range(k)] for _ in values],
            score=[[1.0 for _ in range(k)] for _ in values],
            k=k,
        )

    def dill_inspect(self) -> bool:
        """check if the module can be pickled."""
        return dill.pickles(self)


class FakeDataset:
    """
    A small drop in replacement for `Dataset` objetcs used as corpus.
    """

    column_names: list = ["question.text", "document.row_idx"]

    def remove_columns(self, *args, **kwargs):
        return self

    def __init__(self):
        self.data = {"document.text": "<text>", "document.row_idx": 0}

    def __getitem__(self, idx):
        """check if the module can be pickled."""
        if isinstance(idx, str):
            return [deepcopy(self.data)]
        else:
            return deepcopy(self.data)

    def select(self, index: List[int], **kwargs):
        values = {k: [deepcopy(v) for _ in index] for k, v in self.data.items()}
        values["document.row_idx"] = index
        values = {k: numpy_to_pyarrow_listarray(v) for k, v in values.items()}
        return pa.table(values)
