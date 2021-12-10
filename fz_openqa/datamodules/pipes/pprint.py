from typing import List
from typing import Optional

import rich
from transformers import PreTrainedTokenizerFast

from .base import Pipe
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.datastruct import Eg
from fz_openqa.utils.pretty import get_separator
from fz_openqa.utils.pretty import pprint_batch


class PrintBatch(Pipe):
    """
    Print the batch
    """

    def __init__(self, header: Optional[str] = None, **kwargs):
        super(PrintBatch, self).__init__(**kwargs)
        self.header = header

    def _call_batch(self, batch: Batch, **kwargs) -> Batch:
        """The call of the pipeline process"""
        header = self.header
        if header is None:
            header = "PrintBatch"
        if self.id is not None:
            header = f"{header} (id={self.id})"
        pprint_batch(batch, header=header)
        if len(kwargs):
            kwargs = {k: type(v) for k, v in kwargs.items() if v is not None}
            rich.print(f"PrintBatch input kwargs = {kwargs}")
        return batch

    def _call_egs(self, examples: List[Eg], **kwargs) -> List[Eg]:
        """The call of the pipeline process"""

        header = f"{self.header} : " if self.header is not None else ""
        try:
            pprint_batch(examples[0], header=f"{header}First example")
        except Exception:
            rich.print(f"#{header}Failed to print using pprint_batch. First Example:")
            rich.print(examples[0])

        return examples


class PrintText(Pipe):
    """
    Print the batch
    """

    def __init__(
        self,
        text_key: str,
        limit: Optional[int] = None,
        tokenizer: Optional[PreTrainedTokenizerFast] = None,
        header: Optional[str] = None,
        **kwargs,
    ):
        super(PrintText, self).__init__(**kwargs)
        self.text_key = text_key
        self.limit = limit
        self.header = header
        self.tokenizer = tokenizer

    def _call_batch(self, batch: Batch, **kwargs) -> Batch:
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
            if self.tokenizer:
                txt = self.tokenizer.decode(txt)
            print(txt)
        print(get_separator())

        return batch
