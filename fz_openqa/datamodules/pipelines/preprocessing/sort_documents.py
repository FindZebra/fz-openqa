from fz_openqa.datamodules.pipes import FilterKeys
from fz_openqa.datamodules.pipes import Nested
from fz_openqa.datamodules.pipes import Sequential
from fz_openqa.datamodules.pipes import Sort
from fz_openqa.datamodules.utils.filter_keys import KeyWithPrefix


class SortDocuments(Sequential):
    # todo: check that this works as expected
    def __init__(self):
        super().__init__(
            FilterKeys(KeyWithPrefix("document.")),
            Nested(
                Sort(
                    keys=["document.match_score", "document.retrieval_score"],
                    reversed=True,
                )
            ),
            id="sort-documents",
        )
