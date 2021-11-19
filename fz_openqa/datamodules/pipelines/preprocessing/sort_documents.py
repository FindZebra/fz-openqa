from fz_openqa.datamodules.pipes import DropKeys
from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.datamodules.pipes import Sequential
from fz_openqa.datamodules.pipes import Sort
from fz_openqa.datamodules.pipes.control.condition import HasPrefix
from fz_openqa.datamodules.pipes.nesting import Nested
from fz_openqa.utils.datastruct import Batch


class GenIsPositive(Pipe):
    def _call_batch(self, batch: Batch, **kwargs) -> Batch:
        batch["document.is_positive"] = [x > 0 for x in batch["document.match_score"]]
        return batch


class SortDocuments(Sequential):
    def __init__(self, level: int = 1):
        super().__init__(
            Nested(
                Sequential(
                    GenIsPositive(),
                    Sort(
                        keys=[
                            "document.is_positive",
                            "document.retrieval_score",
                        ],
                        reverse=True,
                    ),
                    DropKeys(["document.is_positive"]),
                ),
                level=level,
            ),
            input_filter=HasPrefix("document."),
            id="sort-documents",
        )
