from datasets import Dataset

from fz_openqa.datamodules.pipes import ApplyAsFlatten
from fz_openqa.datamodules.pipes import FilterKeys
from fz_openqa.datamodules.pipes import RelevanceClassifier
from fz_openqa.datamodules.pipes import Sequential
from fz_openqa.datamodules.pipes.control.condition import In
from fz_openqa.datamodules.pipes.search import FetchDocuments


class ClassifyDocuments(Sequential):
    def __init__(
        self,
        corpus_dataset: Dataset,
        relevance_classifier: RelevanceClassifier,
    ):
        super().__init__(
            ApplyAsFlatten(
                FetchDocuments(
                    corpus_dataset=corpus_dataset,
                    keys=["document.text"],
                ),
                input_filter=In(["document.row_idx"]),
                update=True,
            ),
            relevance_classifier,
            input_filter=In(
                [
                    "document.row_idx",
                    "answer.text",
                    "answer.target",
                ]
            ),
            id="classify-documents",
        )
