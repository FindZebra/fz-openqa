import abc
from typing import List
from typing import Optional
from typing import Union

import spacy
from spacy.tokens import Doc

from .base import Pipe
from fz_openqa.datamodules.pipes.utils.static import STOP_WORDS
from fz_openqa.utils.datastruct import Batch


class TextFilter(Pipe):
    __metaclass__ = abc.ABCMeta

    def __init__(self, *, text_key: Union[str, List[str]], **kwargs):
        super(TextFilter, self).__init__(**kwargs)
        self.text_key = text_key

    def _call_batch(self, batch: Batch, text_key: Optional[str] = None, **kwargs) -> Batch:
        text_key = text_key or self.text_key
        if isinstance(text_key, str):
            text_key = [text_key]

        output = {}
        for key in text_key:
            if key in batch.keys():
                output[key] = self.filter_batch(batch[key])
        return output

    def filter_batch(self, texts: List[str]) -> List[str]:
        return [self.filter_one(text) for text in texts]

    @abc.abstractmethod
    def filter_one(self, text: str) -> str:
        raise NotImplementedError


class StopWordsFilter(TextFilter):
    """Example: remove stop words from string"""

    def filter_one(self, text: str) -> str:
        return " ".join([word for word in text.split() if word.lower() not in STOP_WORDS])


class SciSpaCyFilter(TextFilter):
    """
    Build a Pipe to return a tuple of displacy image of named or
    unnamed word entities and a set of unique entities recognized
    based on scispacy model in use

    Parameters
    ----------
    model_name:
        Name of a pretrained model from spaCy or scispaCy
    text_key:
        text key to be analysed
    """

    no_fingerprint = ["model"]

    def __init__(self, model_name: str = "en_core_sci_lg", **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.model = None

    def __repr__(self):
        return f"SciSpaCyFilter(model_name={self.model_name})"

    def _load_model(self):
        return spacy.load(
            self.model_name,
            disable=[
                "tok2vec",
                "tagger",
                "parser",
                "attribute_ruler",
                "lemmatizer",
            ],
        )

    def filter_one(self, text: str) -> str:
        doc = self.model.pipe(text)
        return self._join_ents(doc)

    def filter_batch(self, texts: List[str]) -> List[str]:
        if self.model is None:
            self.model = self._load_model()
        docs = self.model.pipe(texts)
        return [self._join_ents(doc) for doc in docs]

    @staticmethod
    def _join_ents(doc: Doc) -> str:
        return ", ".join([str(ent.text) for ent in doc.ents])

    def __getstate__(self):
        state = self.__dict__.copy()
        state["model"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.model = None
