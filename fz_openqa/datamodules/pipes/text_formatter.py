import re
from typing import Callable
from typing import List
from typing import Optional
from typing import Union

from .base import Pipe
from fz_openqa.utils.datastruct import Batch


class TextFormatter(Pipe):
    """clean the text field (lower case, apply regex to remove special characters)"""

    def __init__(
        self,
        text_key: Optional[Union[str, List[str]]] = None,
        *,
        remove_breaks: bool = True,
        remove_ref: bool = True,
        remove_hex: bool = True,
        lowercase: bool = False,
        aggressive_cleaning: bool = False,
        remove_symbols: bool = False,
        update: bool = True,
        **kwargs,
    ):
        super().__init__(update=update, **kwargs)
        self.text_key = text_key
        self.remove_breaks = remove_breaks
        self.remove_ref = remove_ref
        self.remove_hex = remove_hex
        self.lowercase = lowercase
        self.aggressive_cleaning = aggressive_cleaning
        self.remove_symbols = remove_symbols

    def clean(self, text: str) -> str:

        if self.lowercase:
            text = text.lower()

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
            # remove "\u" followed by 4 characters
            text = re.sub(r"\\u[0-9a-fA-F]{4}", "", text)
            # remove scientific citations (e.g. (Smith, J. et al., 2014) )
            text = re.sub(r"\((?:[\w \.&]+\, )+[0-9]{4}\)", "", text)
            # remove figure and table references (e.g. (figure 7\xe2\x80\x9377) )
            text = re.sub(r"\s*\(\s*(?:table|figure)[^()]*\)", "", text)

        if self.remove_hex:
            # remove hex characters (e.g. \xe2\x80\x94\xe2\x80\x89)
            text = re.sub(r"[^\x00-\x7f]+", " ", text)

        if self.remove_breaks:
            text = re.sub(r"[\n\r\t]+", " ", text)

        if self.remove_symbols:
            text = re.sub(r"([^a-zA-Z0-9\.])", " ", text)

        return text

    @staticmethod
    def _apply_to_leaves(x: Union[str, List], fn: Callable):
        if isinstance(x, str):
            return fn(x)
        elif isinstance(x, (tuple, list)):
            return [TextFormatter._apply_to_leaves(y, fn) for y in x]
        else:
            ValueError(f"Cannot handle type {type(x).__name__}.")

    def _call_batch(self, batch: Batch, text_key: Optional[str] = None, **kwargs) -> Batch:
        text_key = text_key or self.text_key
        assert text_key is not None, "attribute `text_key` must be set."
        if isinstance(text_key, str):
            text_key = [text_key]
        return {key: self._apply_to_leaves(batch[key], self.clean) for key in text_key}
