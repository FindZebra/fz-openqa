from datasets import Dataset

from fz_openqa.datamodules.pipes import AsFlatten
from fz_openqa.datamodules.pipes import FilterKeys
from fz_openqa.datamodules.pipes import PrintBatch
from fz_openqa.datamodules.pipes import RelevanceClassifier
from fz_openqa.datamodules.pipes import Sequential
from fz_openqa.datamodules.pipes import UpdateWith
from fz_openqa.datamodules.pipes.search import FakeDataset
from fz_openqa.datamodules.pipes.search import FeatchDocuments
from fz_openqa.datamodules.utils.filter_keys import KeyIn


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
            UpdateWith(
                Sequential(
                    FilterKeys(KeyIn(["document.row_idx"])),
                    AsFlatten(
                        FeatchDocuments(
                            corpus_dataset=corpus_dataset,
                            keys=["document.text"],
                        )
                    ),
                )
            ),
            FilterKeys(KeyIn(classifier_input_keys)),
            relevance_classifier,
            id="classify-documents",
        )
