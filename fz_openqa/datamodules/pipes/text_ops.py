import re
from typing import Optional

from .base import Pipe
from fz_openqa.utils.datastruct import Batch


class TextCleaner(Pipe):
    """clean the text field (lower case, apply regex to remove special characters)"""

    def __init__(
        self,
        text_key: Optional[str] = None,
        *,
        remove_linebreaks: bool = True,
        remove_ref: bool = True,
        lowercase: bool = False,
        aggressive_cleaning: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.text_key = text_key
        self.remove_linebreaks = remove_linebreaks
        self.remove_ref = remove_ref
        self.lowercase = lowercase
        self.aggressive_cleaning = aggressive_cleaning

    def set_text_key(self, text_key: str):
        self.text_key = text_key

    def clean(self, text: str) -> str:

        if self.lowercase:
            text = text.lower()

        if self.remove_linebreaks:
            text = re.sub(r"[\n]", "", text)

        if self.aggressive_cleaning:
            # quick and dirty fix (trying to solve issue #80), doesn't fix it
            # todo: do this in a more principled and efficient way
            text = text.replace(r'"', "")
            text = re.compile(r"\\(.)").sub(r"\1", text)
            text = text.strip().replace(r"\n", " ").replace("\n", " ")
            text = text.strip().replace(r"\t", " ").replace("\t", " ")
            text = re.sub(r"[^a-zA-Z ]+", " ", text)
            text = re.sub(" +", " ", text)

        if self.remove_ref:
            text = re.sub(r"\u2003", " ", text)

        return text

    def __call__(
        self, batch: Batch, text_key: Optional[str] = None, **kwargs
    ) -> Batch:
        text_key = text_key or self.text_key
        assert text_key is not None
        batch[text_key] = [self.clean(txt) for txt in batch[text_key]]
        return batch
