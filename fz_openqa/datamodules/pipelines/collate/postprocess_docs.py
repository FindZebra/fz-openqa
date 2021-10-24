from typing import Optional

from fz_openqa.datamodules.pipes import BlockSequential
from fz_openqa.datamodules.pipes import Gate
from fz_openqa.datamodules.pipes import Identity
from fz_openqa.datamodules.pipes import Nested
from fz_openqa.datamodules.pipes import PrintBatch
from fz_openqa.datamodules.pipes import RelevanceClassifier
from fz_openqa.datamodules.pipes import SelectDocs
from fz_openqa.datamodules.pipes import Sequential
from fz_openqa.datamodules.pipes import Sort
from fz_openqa.datamodules.pipes.documents import ARE_DOCS_SELECTED_KEY
from fz_openqa.datamodules.utils.condition import HasKeyWithPrefix
from fz_openqa.datamodules.utils.condition import Not
from fz_openqa.datamodules.utils.condition import Reduce
from fz_openqa.datamodules.utils.condition import Static
from fz_openqa.datamodules.utils.filter_keys import KeyWithPrefix


class PostprocessPipe(BlockSequential):
    def __init__(
        self,
        relevance_classifier: RelevanceClassifier,
        *,
        n_retrieved_documents: int,
        n_select_documents: Optional[int],
        max_select_pos_docs: Optional[int],
        **kwargs
    ):
        """Get the pipe that classify documents as `match_score`,
        sort them and select `n_documents` among the `n_retrieved_docs`"""
        if relevance_classifier is None:
            super().__init__([("identity", Identity())])
        else:
            # sort the documents based on score and `match_score`
            sorter = Nested(
                Sequential(
                    Sort(key="document.retrieval_score"),
                    Sort(key="document.match_score"),
                ),
                filter=KeyWithPrefix("document."),
            )

            # condition to activate the relevance classifier
            activate_doc_proc = Reduce(
                Static(n_retrieved_documents > 0),
                HasKeyWithPrefix("document."),
                Not(HasKeyWithPrefix("document.match_score")),
                reduce_op=all,
            )
            classify_and_sort = Gate(
                activate_doc_proc,
                Sequential(relevance_classifier, sorter),
            )

            # select `n_documents` where count(match_score) <= max_pos_docs
            # this is only ran if the docs have not been previously selected
            selector = SelectDocs(
                total=n_select_documents,
                max_pos_docs=max_select_pos_docs,
                strict=False,
            )
            activate_selector = Reduce(
                HasKeyWithPrefix("document.match_score"),
                Not(HasKeyWithPrefix(ARE_DOCS_SELECTED_KEY)),
                reduce_op=all,
            )
            selector = Gate(activate_selector, selector)

            super().__init__(
                [
                    ("Classify and sort documents", classify_and_sort),
                    ("select documents", selector),
                ],
                **kwargs
            )
