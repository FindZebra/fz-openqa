from datasets import Dataset

from fz_openqa.datamodules.index.index_pipes import FetchDocuments
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


class FetchAndClassifyDocuments(Sequential):
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
        """
        Parameters
        ----------
        corpus_dataset
            The corpus dataset from which to retrieve the documents
        classifier
            The classifier to use
        axis
            The axis to expand on
        n
            The number of documents to expand
        level
            The nesting level to consider to apply the classifier
        extract_gold
            Whether to extract the gold answer from the answer options
        kwargs
            Additional keyword arguments to pass to the Sequential constructor
        """

        super().__init__(
            ApplyAsFlatten(
                FetchDocuments(
                    corpus_dataset=corpus_dataset, keys=[f"{classifier.document_field}.text"]
                ),
                input_filter=In([f"{classifier.document_field}.row_idx"]),
                update=True,
                level=level,
            ),
            ExpandAndClassify(classifier, axis=axis, n=n, level=level, extract_gold=extract_gold),
            input_filter=In(
                [
                    f"{classifier.document_field}.row_idx",
                    f"{classifier.answer_field}.text",
                    f"{classifier.answer_field}.target",
                ]
            ),
        )


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
        """
        Parameters
        ----------
        classifier
            The classifier to use
        axis
            The axis to expand on
        n
            The number of documents to expand
        level
            The nesting level to consider to apply the classifier
        extract_gold
            Whether to extract the gold answer from the answer options
        kwargs
            Additional keyword arguments to pass to the Sequential constructor
        """

        # input filter to this pipeline
        input_filter = Reduce(
            HasPrefix(classifier.document_field), HasPrefix(classifier.answer_field), reduce_op=any
        )

        # extract the gold answer using the `text` and `target` keys
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

        # keys to expand
        expand_targets = [f"{classifier.answer_field}.{k}" for k in ["text", "synonyms"]]

        # initialize the Pipeline
        super().__init__(
            extract_pipe,
            FilterKeys(Not(In([f"{classifier.answer_field}.target"]))),
            Expand(axis=axis, n=n, input_filter=In(expand_targets), update=True),
            ApplyAsFlatten(classifier, level=level),
            input_filter=input_filter,
            **kwargs,
        )
