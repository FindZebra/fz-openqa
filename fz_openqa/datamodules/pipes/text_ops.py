from ...utils.datastruct import Batch
from .base import Pipe


class TextCleaner(Pipe):
    def __init__(
        self, text_key: str, *, remove_breakline: bool = True, **kwargs
    ):
        super().__init__(**kwargs)
        self.text_key = text_key
        self.remove_breakline = remove_breakline

    def process_one(self, txt: str) -> str:
        if self.remove_breakline:
            txt = txt.replace("\n", "")
        return txt

    def __call__(self, batch: Batch) -> Batch:
        batch[self.text_key] = [
            self.process_one(txt) for txt in batch[self.text_key]
        ]
        return batch
