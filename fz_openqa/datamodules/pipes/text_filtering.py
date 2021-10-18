from typing import Optional

import spacy
from spacy.tokens import Doc

from .base import Pipe
from .static import STOP_WORDS
from fz_openqa.utils.datastruct import Batch


class TextFilter(Pipe):
    def __init__(self, *, text_key: str):
        self.text_key = text_key

    def __call__(
        self, batch: Batch, text_key: Optional[str] = None, **kwargs
    ) -> Batch:
        text_key = text_key or self.text_key
        batch[text_key] = [self.filter_one(eg) for eg in batch[text_key]]
        return batch

    def filter_one(self, text: str) -> str:
        raise NotImplementedError


class StopWordsFilter(TextFilter):
    """Example: remove stop words from string"""

    def filter_one(self, text: str) -> str:
        return " ".join(
            [word for word in text.split() if word not in STOP_WORDS]
        )


class SciSpacyFilter(TextFilter):
    """
    Build a Pipe to return a tuple of displacy image of named or unnamed word entities and a set of unique entities recognized based on scispacy model in use
    Args:
        model: A pretrained model from spaCy or scispaCy
        document: text data to be analysed
    """

    def __init__(self, spacy_model=None, **kwargs):
        super().__init__(**kwargs)

        if spacy_model is None:
            self.model = spacy.load(
                "en_core_sci_sm",
                disable=[
                    "tok2vec",
                    "tagger",
                    "parser",
                    "attribute_ruler",
                    "lemmatizer",
                ],
            )

        else:
            self.model = spacy_model.load()

    def filter_one(self, text: str) -> str:
        doc = self.model.pipe(text)
        return self._join_ents(doc)

    @staticmethod
    def _join_ents(doc: Doc) -> str:
        return " ".join([str(ent.text) for ent in doc.ents])

    def __call__(
        self, batch: Batch, text_key: Optional[str] = None, **kwargs
    ) -> Batch:
        text_key = text_key or self.text_key
        docs = self.model.pipe(batch[text_key])
        batch[text_key] = [self._join_ents(doc) for doc in docs]
        return batch


class MetaMapFilter(TextFilter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def filter_one(self, text: str) -> str:
        raise NotImplementedError
