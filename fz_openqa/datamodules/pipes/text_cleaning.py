import re
from typing import Optional

from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.utils.datastruct import Batch


class TextCleaner(Pipe):
    """clean the text field (lower case, apply regex to remove special characters)"""

    def __init__(
        self,
        text_key: str,
        lowercase: bool = True,
        aggressive_cleaning: bool = True,
    ):
        self.text_key = text_key
        self.lowercase = lowercase
        self.aggressive_cleaning = aggressive_cleaning

    def clean(self, text: str) -> str:
        if self.lowercase:
            text = text.lower()

        if self.aggressive_cleaning:
            # quick and dirty fix
            # todo: do this in a more principled and efficient way
            text = text.replace(r'"', "")
            text = re.compile(r"\\(.)").sub(r"\1", text)
            text = text.strip().replace(r"\n", " ").replace("\n", " ")
            text = text.strip().replace(r"\t", " ").replace("\t", " ")
            text = re.sub(r"[^a-zA-Z ]+", " ", text)
            text = re.sub(" +", " ", text)

        return text

    def __call__(
        self, batch: Batch, text_key: Optional[str] = None, **kwargs
    ) -> Batch:

        text_key = text_key or self.text_key
        batch[text_key] = [self.clean(t) for t in batch[text_key]]
        return batch
