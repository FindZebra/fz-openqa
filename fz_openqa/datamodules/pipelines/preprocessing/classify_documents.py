from datasets import Dataset

from fz_openqa.datamodules.pipes import ApplyAsFlatten
from fz_openqa.datamodules.pipes import FilterKeys
from fz_openqa.datamodules.pipes import RelevanceClassifier
from fz_openqa.datamodules.pipes import Sequential
from fz_openqa.datamodules.pipes.control.filter_keys import KeyIn
from fz_openqa.datamodules.pipes.search import FeatchDocuments


class ClassifyDocuments(Sequential):
    def __init__(
        self,
        corpus_dataset: Dataset,
        relevance_classifier: RelevanceClassifier,
    ):
        # define the input keys
        input_keys = [
            "document.row_idx",
            "answer.text",
            "answer.target",
        ]
        # define the output keys
        classifier_input_keys = [
            "document.text",
            "answer.text",
            "answer.target",
        ]

        super().__init__(
            FilterKeys(KeyIn(input_keys)),
            ApplyAsFlatten(
                FeatchDocuments(
                    corpus_dataset=corpus_dataset,
                    keys=["document.text"],
                ),
                filter=KeyIn(["document.row_idx"]),
                update=True,
            ),
            FilterKeys(KeyIn(classifier_input_keys)),
            relevance_classifier,
            id="classify-documents",
        )
