from datasets import Dataset

from fz_openqa.datamodules.pipes import AsFlatten
from fz_openqa.datamodules.pipes import CopyBatch
from fz_openqa.datamodules.pipes import FilterKeys
from fz_openqa.datamodules.pipes import RelevanceClassifier
from fz_openqa.datamodules.pipes import Sequential
from fz_openqa.datamodules.pipes import UpdateWith
from fz_openqa.datamodules.pipes.search import FeatchDocuments
from fz_openqa.datamodules.utils.filter_keys import KeyIn


class ClassifyDocuments(Sequential):
    def __init__(
        self, dataset: Dataset, relevance_classifier: RelevanceClassifier
    ):
        # define the input keys
        input_keys = [
            "document.row_idx",
            "question.text",
            "answer.text",
            "answer.target",
        ]
        # define the output keys
        output_keys = [
            "document.text",
            "document.question.text",
            "answer.text",
            "answer.target",
        ]
        # Whole pipe
        super().__init__(
            FilterKeys(KeyIn(input_keys)),
            UpdateWith(
                Sequential(
                    FilterKeys(KeyIn(["document.row_idx"])),
                    AsFlatten(
                        FeatchDocuments(
                            dataset=dataset, keys=["document.text"]
                        )
                    ),
                )
            ),
            FilterKeys(KeyIn(output_keys)),
            relevance_classifier,
            FilterKeys(KeyIn(["document.match_on", "document.match_score"])),
            id="classify-documents",
        )
