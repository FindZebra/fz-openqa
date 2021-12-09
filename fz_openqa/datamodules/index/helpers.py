from copy import deepcopy

import dill

from fz_openqa.datamodules.index.pipes import SearchResult
from fz_openqa.utils.datastruct import Batch


class FakeIndex:
    """A small class to test Search corpus without using a proper index"""

    index_name = "<name>"

    def search(self, *, query: Batch, k: int, **kwargs) -> SearchResult:
        values = query["question.text"]
        return SearchResult(
            index=[[0 for _ in range(k)] for _ in values],
            score=[[1.0 for _ in range(k)] for _ in values],
        )

    def dill_inspect(self) -> bool:
        """check if the module can be pickled."""
        return dill.pickles(self)


class FakeDataset:
    """A small class to test Search corpus without using a proper index"""

    def __init__(self):
        self.data = {"document.text": "<text>", "document.row_idx": 0}

    def __getitem__(self, idx):
        """check if the module can be pickled."""
        if isinstance(idx, str):
            return [deepcopy(self.data)]
        else:
            return deepcopy(self.data)
