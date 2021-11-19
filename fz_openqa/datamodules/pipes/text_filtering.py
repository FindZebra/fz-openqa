import abc
from typing import List
from typing import Optional
from typing import Union

import rich
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


class SciSpacyFilter(TextFilter):
    """
    Build a Pipe to return a tuple of displacy image of named or
    unnamed word entities and a set of unique entities recognized
    based on scispacy model in use
    Args:
        model: A pretrained model from spaCy or scispaCy
        document: text data to be analysed
    """

    def __init__(self, model_name: str = "en_core_sci_sm", **kwargs):
        super().__init__(**kwargs)
        self.model = spacy.load(
            model_name,
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
        docs = self.model.pipe(texts)
        return [self._join_ents(doc) for doc in docs]

    @staticmethod
    def _join_ents(doc: Doc) -> str:
        return " ".join([str(ent.text) for ent in doc.ents])


class MetaMapFilter(TextFilter):
    """
    Build a Pipe to return a string of unique entities recognized
    based on offline processed MetaMap heuristic


    Args:
        MetaMapList: A list of recognised entities inferred from the question query
        Question: query to be replaced by MetaMapList
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # todo: fix this. this cannot work as it is
        raise NotImplementedError

    @staticmethod
    def _join_ents(MetaMapList: list) -> str:
        return " ".join([str(ent) for ent in MetaMapList])

    def _call_batch(self, batch: Batch, query_key: Optional[str] = None, **kwargs) -> Batch:
        rich.print(f"[green]{batch.keys()}")
        query_key = query_key or self.query_key
        batch[query_key] = [self._join_ents(lst) for lst in batch["question.metamap"]]
