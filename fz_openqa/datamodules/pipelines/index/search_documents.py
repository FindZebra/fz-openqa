from datasets import Dataset

from fz_openqa.datamodules.pipes import ApplyAsFlatten
from fz_openqa.datamodules.pipes import FilterKeys
from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.datamodules.pipes import PrintBatch
from fz_openqa.datamodules.pipes import SearchCorpus
from fz_openqa.datamodules.pipes import Sequential
from fz_openqa.datamodules.pipes.control.condition import Contains
from fz_openqa.datamodules.pipes.control.condition import HasPrefix
from fz_openqa.datamodules.pipes.control.condition import In
from fz_openqa.datamodules.pipes.search import FetchDocuments


class FetchNestedDocuments(ApplyAsFlatten):
    """Retrieve the full document rows (text, input_ids, ...) from
    the corpus object given the input `index_key` for nested documents ([[input_ids]])"""

    def __init__(
        self,
        corpus_dataset: Dataset,
        collate_pipe: Pipe,
        update: bool = True,
        index_key: str = "document.row_idx",
    ):
        super().__init__(
            pipe=FetchDocuments(
                corpus_dataset=corpus_dataset,
                collate_pipe=collate_pipe,
            ),
            input_filter=In([index_key]),
            update=update,
        )


class SearchDocuments(Sequential):
    def __init__(self, *, corpus_index, n_documents: int):
        super().__init__(
            SearchCorpus(corpus_index=corpus_index, k=n_documents),
            FilterKeys(In(["document.row_idx", "document.retrieval_score"])),
            id="search-documents",
            input_filter=In(["question.input_ids", "question.attention_mask", "question.text"]),
        )
