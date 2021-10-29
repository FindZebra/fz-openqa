from typing import Final

from fz_openqa.datamodules.pipes import FilterKeys
from fz_openqa.datamodules.pipes import SearchCorpus
from fz_openqa.datamodules.pipes import Sequential
from fz_openqa.datamodules.pipes.control.filter_keys import KeyIn


class SearchDocuments(Sequential):
    def __init__(self, *, corpus_index, n_documents: int):
        super().__init__(
            SearchCorpus(corpus_index=corpus_index, k=n_documents),
            FilterKeys(
                KeyIn(["document.row_idx", "document.retrieval_score"])
            ),
            id="search-documents",
        )
