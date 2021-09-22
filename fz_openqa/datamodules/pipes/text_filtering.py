from ...utils.datastruct import Batch
from .base import Pipe


class TextFilter(Pipe):
    def __init__(self, *, text_key: str):
        self.text_key = text_key

    def __call__(self, batch: Batch) -> Batch:
        raise NotImplementedError


class SciSpacyFilter(TextFilter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, batch):
        raise NotImplementedError


class MetaMapFilter(TextFilter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, batch):
        raise NotImplementedError
