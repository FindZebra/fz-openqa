import math
import re
from typing import Callable
from typing import List
from typing import Optional
from typing import Union

import html2text
from warp_pipes import Batch
from warp_pipes import Pipe


class TextFormatter(Pipe):
    """clean the text field (lower case, apply regex to remove special characters)"""

    def __init__(
        self,
        text_key: Optional[Union[str, List[str]]] = None,
        *,
        remove_breaks: bool = False,
        remove_ref: bool = False,
        lowercase: bool = False,
        aggressive_cleaning: bool = False,
        remove_symbols: bool = False,
        remove_hex: bool = False,
        update: bool = True,
        **kwargs,
    ):
        super().__init__(update=update, **kwargs)
        self.text_key = text_key
        self.remove_breaks = remove_breaks
        self.remove_ref = remove_ref
        self.lowercase = lowercase
        self.aggressive_cleaning = aggressive_cleaning
        self.remove_symbols = remove_symbols
        self.remove_hex = remove_hex

    def clean(self, text: str) -> str:

        if self.lowercase:
            text = text.lower()

        if self.aggressive_cleaning:
            # quick and dirty fix (trying to solve issue #80), doesn't fix it
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

    def _call_batch(
        self,
        batch: Batch,
        idx: Optional[List[int]] = None,
        *,
        text_key: Optional[str] = None,
        **kwargs,
    ) -> Batch:
        text_key = text_key or self.text_key
        assert text_key is not None, "attribute `text_key` must be set."
        if isinstance(text_key, str):
            text_key = [text_key]

        return {
            key: self._apply_to_leaves(batch[key], self.clean)
            for key in text_key
            if key in batch.keys()
        }


class MedQaTextFormatter(TextFormatter):
    def __init__(self, text_key: Optional[Union[str, List[str]]] = None, **kwargs):
        super().__init__(**kwargs)
        self.text_key = text_key

    def clean(self, text: str) -> str:
        text = re.sub(r"([^a-zA-Z0-9\.])", " ", text).strip()
        return text


class HtmlCleaner(TextFormatter):
    def __init__(self, text_key: Optional[Union[str, List[str]]] = None, **kwargs):
        super().__init__(**kwargs)
        self.text_key = text_key

        self.text_maker = html2text.HTML2Text(bodywidth=math.inf)
        self.text_maker.ignore_links = True
        self.text_maker.ignore_images = False
        self.text_maker.ignore_emphasis = True
        self.text_maker.ignore_tables = False
        self.text_maker.images_to_alt = True
        self.text_maker.default_image_alt = "figure"

        self.remove_pattern = re.compile("|".join([r"#+ ", r"\[edit\]"]))

    def clean(self, text: str) -> str:
        text = self.text_maker.handle(text)
        text = self.remove_pattern.sub("", text)
        return text


class ReSubPatternFormatter(TextFormatter):
    def __init__(self, clean_pattern: str = r"(<.*?>)|(\[.*?\])", **kwargs):
        super(ReSubPatternFormatter, self).__init__(**kwargs)
        self.cleanr = re.compile(clean_pattern)

    def clean(self, text: str) -> str:
        return re.sub(self.cleanr, "", text)
