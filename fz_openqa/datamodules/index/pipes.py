from typing import List
from typing import Optional

from datasets import Dataset

from fz_openqa.datamodules.index.base import Index
from fz_openqa.datamodules.index.base import IndexMode
from fz_openqa.datamodules.pipes import ApplyAsFlatten
from fz_openqa.datamodules.pipes.base import Pipe
from fz_openqa.datamodules.pipes.collate import Collate
from fz_openqa.datamodules.pipes.control.condition import In
from fz_openqa.utils.array import concat_arrays
from fz_openqa.utils.datastruct import Batch


class SearchCorpus(ApplyAsFlatten):
    def __init__(
        self,
        index: Index,
        *,
        level: int = 0,
        **kwargs,
    ):
        assert "input_filter" not in kwargs
        self.index = index
        input_filter = In(index.input_keys(IndexMode.QUERY))
        pipe = SearchCorpusFlat(index, **kwargs)
        super().__init__(pipe, level=level, input_filter=input_filter, update_idx=True)


class SearchCorpusFlat(Pipe):
    """Search a Corpus object given a query"""

    def __init__(
        self,
        index: Index,
        *,
        k: Optional[int] = None,
        index_output_key: str = "document.row_idx",
        score_output_key: str = "document.retrieval_score",
        analyzed_output_key: str = "document.analyzed_tokens",
        **kwargs,
    ):
        super(SearchCorpusFlat, self).__init__(**kwargs)
        self.index = index
        self.index_output_key = index_output_key
        self.score_output_key = score_output_key
        self.analyzed_output_key = analyzed_output_key
        self.k = k

    def _call_batch(
        self,
        query: Batch,
        *,
        k: Optional[int] = None,
        **kwargs,
    ):
        # update args
        k = k or self.k

        # query the index
        search_result = self.index.search(query, k=k, **kwargs)

        # store as a dictionary and return
        output = {
            self.index_output_key: search_result.index,
            self.score_output_key: search_result.score,
        }

        if search_result.tokens is not None:
            output[self.analyzed_output_key] = search_result.tokens

        return output


class FetchDocuments(Pipe):
    def __init__(
        self,
        *,
        corpus_dataset: Dataset,
        keys: Optional[List[str]] = None,
        collate_pipe: Pipe = None,
        index_key: str = "document.row_idx",
        id: str = "fetch-documents-pipe",
        **kwargs,
    ):
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

    def output_keys(self, input_keys: List[str]) -> List[str]:
        return self.corpus_dataset.column_names

    def _call_batch(self, batch: Batch, max_chunk_size: int = 500, **kwargs) -> Batch:
        # todo: check dataset fingerprint (checking 1st index for now)

        # get the `dataset` indexes
        # todo: query dataset for unique indexes only {i: idx for i, idx in enumerate(indexes)}
        indexes = [int(idx) for idx in batch[self.index_key]]

        rows = self._fetch_rows(indexes, max_chunk_size=max_chunk_size)
        new_indexes = rows[self.index_key]
        if not new_indexes == indexes:
            raise ValueError(
                f"The number of returned rows does not match with the input index. "
                f"Retrieved {len(new_indexes)} indexes, expected {len(indexes)}."
                f"First 10 retrieved indexes: {new_indexes[:10]}. "
                f"First 10 indexes: {indexes[:10]}. "
                f"Try using a smaller batch size."
            )

        # collate and return
        return self.collate_pipe(rows)

    def _fetch_rows(self, indexes: List[int], max_chunk_size: int = 500) -> Batch:
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
            table = self.corpus_dataset.select(index_i, keep_in_memory=True)
            batch: Batch = table[None:None]
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
