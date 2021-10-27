from fz_openqa.datamodules.corpus_dm import CorpusDataModule
from fz_openqa.datamodules.pipes import FilterKeys
from fz_openqa.datamodules.pipes import SearchCorpus
from fz_openqa.datamodules.pipes import Sequential
from fz_openqa.datamodules.utils.filter_keys import KeyIn


class SearchDocuments(Sequential):
    def __init__(self, *, corpus: CorpusDataModule, n_documents: int):
        super().__init__(
            SearchCorpus(corpus, k=n_documents),
            FilterKeys(
                KeyIn(["document.row_idx", "document.retrieval_score"])
            ),
            id="search-documents",
        )