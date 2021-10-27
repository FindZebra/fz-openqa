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
        text_key: Optional[str] = None,
        *,
        remove_linebreaks: bool = True,
        remove_ref: bool = True,
        lowercase: bool = False,
        aggressive_cleaning: bool = False,
        medqa_cleaning: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.text_key = text_key
        self.remove_linebreaks = remove_linebreaks
        self.remove_ref = remove_ref
        self.lowercase = lowercase
        self.aggressive_cleaning = aggressive_cleaning
        self.medqa_cleaning = medqa_cleaning

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

        if self.medqa_cleaning:
            text = re.sub(r'([^a-zA-Z0-9\.])', " ", text).strip()


        if self.remove_ref:
            text = re.sub(r"\u2003", " ", text)

        return text

    @staticmethod
    def _apply_to_leaves(x: Union[str, List], fn: Callable):
        if isinstance(x, str):
            return fn(x)
        elif isinstance(x, (tuple, list)):
            return [TextFormatter._apply_to_leaves(y, fn) for y in x]
        else:
            ValueError(f"Cannot handle type {type(x).__name__}.")

    def __call__(
        self, batch: Batch, text_key: Optional[str] = None, **kwargs
    ) -> Batch:
        text_key = text_key or self.text_key
        assert text_key is not None, "attribute `text_key` must be set."
        batch[text_key] = self._apply_to_leaves(batch[text_key], self.clean)
        return batch
