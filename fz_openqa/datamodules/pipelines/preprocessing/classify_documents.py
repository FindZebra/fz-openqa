from datasets import Dataset
from warp_pipes import ApplyAsFlatten
from warp_pipes import Expand
from warp_pipes import FilterKeys
from warp_pipes import Sequential
from warp_pipes.core.condition import HasPrefix
from warp_pipes.core.condition import In
from warp_pipes.core.condition import Not
from warp_pipes.core.condition import Reduce

from fz_openqa.datamodules.pipes import ExtractGoldAnswer
from fz_openqa.datamodules.pipes.fetch import FetchDocuments


class FetchDocumentsAndExtractAnswer(Sequential):
    """Retriever the document text field from the corpus and
    classify each document using the answer."""

    def __init__(
        self,
        corpus_dataset: Dataset,
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
                    dataset=corpus_dataset,
                    keys=["document.text"],
                ),
                input_filter=In(["document.row_idx"]),
                update=True,
                level=level,
            ),
            ExpandAndExtractAnswer(axis=axis, n=n, extract_gold=extract_gold),
            input_filter=In(
                [
                    "document.row_idx",
                    "answer.text",
                    "answer.target",
                    "answer.idx",
                ]
            ),
        )


class ExpandAndExtractAnswer(Sequential):
    """Expand the answer field to match the document field and classify"""

    def __init__(
        self,
        *,
        axis: int = 1,
        n: int,
        extract_gold: bool = True,
        **kwargs,
    ):

        # input filter to this pipeline
        input_filter = Reduce(HasPrefix("document"), HasPrefix("answer"), reduce_op=any)

        # extract the gold answer using the `text` and `target` keys
        if extract_gold:
            extract_pipe = ExtractGoldAnswer(
                answer_field="answer",
                options_key="text",
                target_key="target",
                output_key="text",
                update=True,
            )
        else:
            extract_pipe = None

        # keys to expand
        expand_targets = [f"answer.{k}" for k in ["text", "synonyms"]]

        # initialize the Pipeline
        super().__init__(
            extract_pipe,
            FilterKeys(Not(In(["answer.target"]))),
            Expand(axis=axis, n=n, input_filter=In(expand_targets), update=True),
            input_filter=input_filter,
            **kwargs,
        )
