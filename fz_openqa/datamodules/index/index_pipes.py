from __future__ import annotations

from typing import List
from typing import Optional

from datasets import Dataset

from fz_openqa.datamodules.index.index import Index
from fz_openqa.datamodules.pipes import ApplyAsFlatten
from fz_openqa.datamodules.pipes import Partial
from fz_openqa.datamodules.pipes.base import Pipe
from fz_openqa.datamodules.pipes.collate import Collate
from fz_openqa.datamodules.pipes.control.condition import In
from fz_openqa.utils.array import concat_arrays
from fz_openqa.utils.datastruct import Batch


class FetchDocuments(Pipe):
    """
    Fetch documents from a Corpus object given a list of row_idx.

    Notes
    -----
    `datasets.Dataset.__getitem__` is used to fetch the documents. It return fewer
    documents than the requested number of documents if `max_chunk_size` is too large.
    Set `max_chunk_size` to a smaller value to avoid this.

    todo: merge this within the Index using MixIn
    """

    def __init__(
        self,
        *,
        corpus_dataset: Dataset,
        keys: Optional[List[str]] = None,
        collate_pipe: Pipe = None,
        index_key: str = "document.row_idx",
        id: str = "fetch-documents-pipe",
        max_chunk_size: int = 500,
        **kwargs,
    ):
        """
        Parameters
        ----------
        corpus_dataset
            The dataset to fetch the documents from.
        keys
            The keys to fetch from the corpus dataset.
        collate_pipe
            The pipe to use to collate the fetched rows into a batch.
        index_key
            The key used as index for the corpus dataset.
        id
            The id of the pipe.
        max_chunk_size
            The maximum number of rows to fetch at once.
        kwargs
            Additional keyword arguments to pass to the collate pipe.
        """
        super(FetchDocuments, self).__init__(id=id)
        if keys is not None:
            keys.append(index_key)
            # make sure to sort the keys to ensure deterministic fingerprinting
            cols_to_drop = [c for c in corpus_dataset.column_names if c not in keys]
            corpus_dataset = corpus_dataset.remove_columns(cols_to_drop)

        self.corpus_dataset = corpus_dataset
        self.keys = keys
        self.collate_pipe = collate_pipe or Collate()
        self.index_key = index_key
        self.max_chunk_size = max_chunk_size

    def output_keys(self, input_keys: List[str]) -> List[str]:
        return self.corpus_dataset.column_names

    def _call_batch(self, batch: Batch, **kwargs) -> Batch:
        # todo: check dataset fingerprint (checking 1st index for now)

        # get the `dataset` indexes
        # todo: query dataset for unique indexes only (torch.unique)
        indexes = [int(idx) for idx in batch[self.index_key]]

        if len(indexes) == 0:
            return {}

        rows = self._fetch_rows(indexes, max_chunk_size=self.max_chunk_size)
        new_indexes = rows[self.index_key]
        if len(new_indexes) != len(indexes):
            raise ValueError(
                f"The number of returned rows does not match with the input index. "
                f"Retrieved {len(new_indexes)} indexes, expected {len(indexes)}."
            )

        no_neg_indices = [i for i in range(len(indexes)) if indexes[i] >= 0]
        if new_indexes[no_neg_indices[0]] != indexes[no_neg_indices[0]]:
            raise ValueError(
                f"The retrieved indices do not matched the query indicies. "
                f"First 10 retrieved indexes: {new_indexes[:10]}. "
                f"First 10 query indexes: {indexes[:10]}. "
                f"Try using a smaller batch size."
            )

        # collate and return
        output = self.collate_pipe(rows)
        return output

    def _fetch_rows(self, indexes: List[int], max_chunk_size: int = 100) -> Batch:
        """
        Fetch rows from the corpus dataset given a list of indexes.

        Notes
        -----
        `Dataset.select` fails when the index is too large. Chunk the indexes to avoid this issue.
        """

        rows = None
        # fetch documents
        for i in range(0, len(indexes), max_chunk_size):
            index_i = indexes[i : i + max_chunk_size]
            batch = self.corpus_dataset[index_i]
            if rows is None:
                rows = batch
            else:
                for k, v in batch.items():
                    rows[k] = concat_arrays(rows[k], v)

        return rows


class FetchNestedDocuments(ApplyAsFlatten):
    """Retrieve the full document rows (text, input_ids, ...) from
    the corpus object given the input `index_key` for nested documents ([[input_ids]])"""

    def __init__(
        self,
        corpus_dataset: Dataset,
        collate_pipe: Pipe,
        update: bool = True,
        index_key: str = "document.row_idx",
        level: int = 1,
    ):
        pipe = FetchDocuments(
            corpus_dataset=corpus_dataset,
            collate_pipe=collate_pipe,
        )

        super().__init__(pipe=pipe, input_filter=In([index_key]), update=update, level=level)
