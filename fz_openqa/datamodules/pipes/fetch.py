from __future__ import annotations

from typing import List
from typing import Optional

from datasets import Dataset
from warp_pipes import ApplyAsFlatten
from warp_pipes import Batch
from warp_pipes import In
from warp_pipes import keep_only_columns
from warp_pipes import Nest
from warp_pipes import Pipe
from warp_pipes.support.tensor_handler import TensorFormat
from warp_pipes.support.tensor_handler import TensorHandler


class FetchDocuments(Pipe):
    """
    Fetch documents from a Dataset object given a list of row_idx.
    """

    def __init__(
        self,
        *,
        dataset: Dataset,
        keys: Optional[List[str]] = None,
        index_key: str = "document.row_idx",
        **kwargs,
    ):
        super(FetchDocuments, self).__init__(**kwargs)
        if keys is not None:
            dataset = keep_only_columns(dataset, keys)

        self.dataset = dataset
        self.index_key = index_key

    def _call_batch(self, batch: Batch, idx: Optional[List[int]] = None, **kwargs) -> Batch:
        # get the indices and flatten them
        handler = TensorHandler(TensorFormat.NUMPY)
        ids = batch[self.index_key]
        ids = handler(ids)
        ref_shape = ids.shape
        ids = ids.reshape(-1)

        # fetch the documents
        fetched_batch = self.dataset[ids]

        # reshape the documents
        if len(ref_shape) > 1:
            nest_pipe = Nest(shape=list(ref_shape))
            fetched_batch = nest_pipe(fetched_batch)

        return fetched_batch


class NestedFetchDocuments(ApplyAsFlatten):
    """Retrieve the full document rows (text, input_ids, ...) from
    the corpus object given the input `index_key` for nested documents ([[input_ids]])"""

    def __init__(
        self,
        dataset: Dataset,
        collate_pipe: Pipe,
        update: bool = True,
        index_key: str = "document.row_idx",
        pprint: bool = False,
        **kwargs,
    ):
        pipe = FetchDocuments(
            dataset=dataset, collate_pipe=collate_pipe, input_filter=In([index_key]), **kwargs
        )

        super().__init__(
            pipe=pipe,
            update=update,
            level=[index_key],
            pprint=pprint,
        )
