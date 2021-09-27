from typing import Optional

from ...utils.datastruct import Batch
from .base import Pipe
from .static import STOP_WORDS


class TextFilter(Pipe):
    def __init__(self, *, text_key: str):
        self.text_key = text_key

    def __call__(
        self, batch: Batch, text_key: Optional[str] = None, **kwargs
    ) -> Batch:
        text_key = text_key or self.text_key
        batch[text_key] = [self.filter(eg) for eg in batch[text_key]]
        return batch

    def filter(self, text: str) -> str:
        raise NotImplementedError


class StopWordsFilter(TextFilter):
    """Example: remove stop words from string"""

    def filter(self, text: str) -> str:
        return " ".join(
            [word for word in text.split() if word not in STOP_WORDS]
        )


class SciSpacyFilter(TextFilter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def filter(self, text: str) -> str:
        raise NotImplementedError


class MetaMapFilter(TextFilter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def filter(self, text: str) -> str:
        raise NotImplementedError
