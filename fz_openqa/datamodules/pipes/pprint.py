from typing import Optional

from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.pretty import get_separator
from fz_openqa.utils.pretty import pprint_batch


class PrintBatch(Pipe):
    """
    Print the batch
    """

    def __init__(self, header: Optional[str] = None):
        self.header = header

    def __call__(self, batch: Batch, **kwargs) -> Batch:
        """The call of the pipeline process"""
        pprint_batch(batch, header=self.header)

        return batch


class PrintText(Pipe):
    """
    Print the batch
    """

    def __init__(
        self,
        text_key: str,
        limit: Optional[int] = None,
        header: Optional[str] = None,
    ):
        self.text_key = text_key
        self.limit = limit
        self.header = header

    def __call__(self, batch: Batch, **kwargs) -> Batch:
        """The call of the pipeline process"""
        txts = batch.get(self.text_key, None)
        if self.limit:
            txts = txts[: self.limit]
        print(get_separator())
        if self.header is None:
            print(f"=== {self.text_key} ===")
        else:
            print(f"=== {self.header} : {self.text_key} ===")
        for txt in txts:
            print(txt)
        print(get_separator())

        return batch
