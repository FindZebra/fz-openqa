from datasets import Dataset

from fz_openqa.datamodules.index.pipes import FetchDocuments
from fz_openqa.datamodules.pipes import ApplyAsFlatten
from fz_openqa.datamodules.pipes import ExtractGoldAnswer
from fz_openqa.datamodules.pipes import FilterKeys
from fz_openqa.datamodules.pipes import RelevanceClassifier
from fz_openqa.datamodules.pipes import Sequential
from fz_openqa.datamodules.pipes.control.condition import HasPrefix
from fz_openqa.datamodules.pipes.control.condition import In
from fz_openqa.datamodules.pipes.control.condition import Not
from fz_openqa.datamodules.pipes.control.condition import Reduce
from fz_openqa.datamodules.pipes.nesting import Expand


class ExpandAndClassify(Sequential):
    """Expand the answer field to match the document field and classify"""

    def __init__(
        self,
        classifier: RelevanceClassifier,
        *,
        axis: int = 1,
        n: int,
        level=1,
        extract_gold: bool = True,
        **kwargs,
    ):
        input_filter = Reduce(
            HasPrefix(classifier.document_field), HasPrefix(classifier.answer_field), reduce_op=any
        )

        if extract_gold:
            extract_pipe = ExtractGoldAnswer(
                answer_field=classifier.answer_field,
                options_key="text",
                target_key="target",
                output_key="text",
                update=True,
            )
        else:
            extract_pipe = None

        super().__init__(
            extract_pipe,
            FilterKeys(Not(In([f"{classifier.answer_field}.target"]))),
            Expand(
                axis=axis, n=n, input_filter=In([f"{classifier.answer_field}.text"]), update=True
            ),
            ApplyAsFlatten(classifier, level=level),
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
        n: int,
        level: int = 1,
        extract_gold: bool = True,
    ):
        super().__init__(
            ApplyAsFlatten(
                Sequential(FetchDocuments(corpus_dataset=corpus_dataset, keys=["document.text"])),
                input_filter=In(["document.row_idx"]),
                update=True,
                level=level,
            ),
            ExpandAndClassify(classifier, axis=axis, n=n, level=level, extract_gold=extract_gold),
            input_filter=In(
                [
                    "document.row_idx",
                    "answer.text",
                    "answer.target",
                ]
            ),
            id="classify-documents",
        )
