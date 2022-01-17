from __future__ import annotations

import logging
from collections import defaultdict
from typing import List
from typing import Optional

import numpy as np
import rich
from datasets import Dataset

from ..utils.dataset import keep_only_columns
from .base import Index
from .base import IndexMode
from .search_result import SearchResult
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.datastruct import OutputFormat

# load the default es configuration

logger = logging.getLogger("fz_openqa.index.static")


class StaticIndex(Index):
    default_key = None
    no_fingerprint: List[str] = Index.no_fingerprint + ["document_lookup"]

    def input_keys(self, mode: Optional[IndexMode] = None) -> List[str]:
        """Return the input keys for the given mode"""
        if mode == IndexMode.INDEX:
            return [f"{self.index_field}.idx"]
        elif mode == IndexMode.QUERY:
            return [f"{self.query_field}.document_idx"]
        else:
            raise ValueError(f"Unknown mode {mode}")

    def _prepare_index(
        self,
        *args,
        **kwargs,
    ):
        pass

    def build(self, dataset: Dataset, verbose: bool = False, **kwargs):
        """Index the dataset using elastic search.
        We make sure a unique index is created for each dataset"""
        # set a unique index name
        self._set_index_name(dataset=dataset)
        dataset = keep_only_columns(dataset, self.input_keys(IndexMode.INDEX))
        self.document_lookup = defaultdict(list)
        for row_idx, doc_id in enumerate(dataset[f"{self.index_field}.idx"]):
            self.document_lookup[doc_id].append(row_idx)

        logger.info(
            f"n_documents={len(self.document_lookup.keys())}, "
            f"n_rows/doc={np.mean([len(rows) for rows in self.document_lookup.values()]):.1f},"
            f"max_rows/doc={np.max([len(rows) for rows in self.document_lookup.values()]):.1f}"
        )

        self.is_indexed = True

    def _preprocess_query(self, batch: Batch, **kwargs) -> Batch:
        """Preprocess the batch before query"""
        return batch

    def _search_chunk(self, query: Batch, *, k: int, **kwargs) -> SearchResult:
        """Search the index for the given query."""
        doc_ids = query[f"{self.query_field}.document_idx"]
        rows_ids = [self.document_lookup[doc_id] for doc_id in doc_ids]

        # build the results
        return SearchResult(
            score=[[1.0 for _ in r] for r in rows_ids],
            index=rows_ids,
            dataset_size=self.dataset_size,
            k=k,
            format=OutputFormat.NUMPY,
        )
