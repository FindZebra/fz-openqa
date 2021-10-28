from fz_openqa.datamodules.pipes import ApplyAsFlatten
from fz_openqa.datamodules.pipes import FilterKeys
from fz_openqa.datamodules.pipes import Sequential
from fz_openqa.datamodules.pipes import Sort
from fz_openqa.datamodules.pipes.nesting import Nested
from fz_openqa.datamodules.utils.filter_keys import KeyWithPrefix


class SortDocuments(Sequential):
    def __init__(self):
        super().__init__(
            Nested(
                Sort(
                    keys=["document.match_score", "document.retrieval_score"],
                    reversed=True,
                ),
                filter=KeyWithPrefix("document."),
            ),
            id="sort-documents",
        )
