import re

from ...utils.datastruct import Batch
from .base import Pipe


class TextCleaner(Pipe):
    def __init__(
        self,
        text_key: str,
        *,
        remove_breakline: bool = True,
        remove_ref: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.text_key = text_key
        self.remove_breakline = remove_breakline
        self.remove_ref = remove_ref

    def process_one(self, txt: str) -> str:
        if self.remove_breakline:
            txt = re.sub(r"[\n]", "", txt)

        if self.remove_ref:
            txt = re.sub(r"\u2003", " ", txt)

        return txt

    def __call__(self, batch: Batch) -> Batch:
        batch[self.text_key] = [
            self.process_one(txt) for txt in batch[self.text_key]
        ]
        return batch
