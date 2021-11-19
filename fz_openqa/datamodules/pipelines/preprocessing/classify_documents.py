from datasets import Dataset

from fz_openqa.datamodules.index.pipes import FetchDocuments
from fz_openqa.datamodules.pipes import ApplyAsFlatten
from fz_openqa.datamodules.pipes import RelevanceClassifier
from fz_openqa.datamodules.pipes import Sequential
from fz_openqa.datamodules.pipes.control.condition import HasPrefix
from fz_openqa.datamodules.pipes.control.condition import In
from fz_openqa.datamodules.pipes.control.condition import Reduce
from fz_openqa.datamodules.pipes.nesting import Expand


class ExpandAndClassify(Sequential):
    """Expand the answer field to match the document field and classify"""

    def __init__(
        self, classifier: RelevanceClassifier, *, axis: int = 1, n_documents: int, **kwargs
    ):
        input_filter = Reduce(
            HasPrefix(classifier.document_field), HasPrefix(classifier.answer_field), reduce_op=any
        )
        super().__init__(
            Expand(
                axis=axis,
                n=n_documents,
                input_filter=HasPrefix(classifier.answer_field),
                update=True,
            ),
            ApplyAsFlatten(classifier, level=1),
            input_filter=input_filter,
            **kwargs,
        )


class ClassifyDocuments(Sequential):
    """Retriever the document text field from the corpus and
    classify each document using the answer."""

    def __init__(
        self,
        corpus_dataset: Dataset,
        classifier: RelevanceClassifier,
        *,
        axis: int = 1,
        n_documents: int,
    ):
        super().__init__(
            ApplyAsFlatten(
                Sequential(FetchDocuments(corpus_dataset=corpus_dataset, keys=["document.text"])),
                input_filter=In(["document.row_idx"]),
                update=True,
            ),
            ExpandAndClassify(classifier, axis=axis, n_documents=n_documents),
            input_filter=In(
                [
                    "document.row_idx",
                    "answer.text",
                    "answer.target",
                ]
            ),
            id="classify-documents",
        )
