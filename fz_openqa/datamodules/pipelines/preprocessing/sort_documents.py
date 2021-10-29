from fz_openqa.datamodules.pipes import Sequential
from fz_openqa.datamodules.pipes import Sort
from fz_openqa.datamodules.pipes.control.filter_keys import KeyWithPrefix
from fz_openqa.datamodules.pipes.nesting import Nested


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
