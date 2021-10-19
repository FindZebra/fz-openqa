import re
from dataclasses import dataclass
from itertools import tee
from itertools import zip_longest
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Sequence

import dill
import numpy as np
import spacy
import torch
from scispacy.abbreviation import AbbreviationDetector  # type: ignore
from scispacy.linking import EntityLinker  # type: ignore
from scispacy.linking_utils import Entity
from spacy import Language
from spacy.tokens import Doc

from ..utils.filter_keys import KeyWithPrefix
from .static import DISCARD_TUIs
from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.utils.datastruct import Batch

np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


@dataclass
class Pair:
    document: Dict[str, Any]
    answer: Dict[str, Any]


def find_one(
    text: str, queries: Sequence[Any], sort_by: Optional[Callable] = None
) -> bool:
    """check if one of the queries is in the input text"""
    assert isinstance(text, str)
    queries = set(queries)
    if len(queries) == 0:
        return False
    if len(text) == 0:
        return False

    if sort_by is not None:
        queries = sorted(queries, key=sort_by)

    # re.search: Scan through string looking for a location where the regular expression pattern produces a match, and return a corresponding MatchObject instance. Return None if no position in the string matches the pattern; note that this is different from finding a zero-length match at some point in the string.
    # re.escape: Return string with all non-alphanumerics backslashed; this is useful if you want to match an arbitrary literal string that may have regular expression metacharacters in it.
    # re.IGNORECASE: Perform case-insensitive matching
    return bool(
        re.search(
            re.compile(
                "|".join(re.escape(x) for x in queries),
                re.IGNORECASE,
            ),
            text,
        )
    )


class RelevanceClassifier(Pipe):
    def __init__(
        self,
        answer_prefix: str = "answer.",
        document_prefix: str = "document.",
        output_key: str = "document.is_positive",
    ):
        self.output_key = output_key
        self.answer_prefix = answer_prefix
        self.document_prefix = document_prefix

    def classify(self, pair: Pair) -> bool:
        """Classify each pair."""
        raise NotImplementedError

    def preprocess(self, pairs: Iterable[Pair]) -> Iterable[Pair]:
        """Preprocessing allows transforming all the pairs,
        potentially in batch mode."""
        return pairs

    def _infer_n_docs(self, batch: Batch) -> int:
        x = [
            v
            for k, v in batch.items()
            if str(k).startswith(self.document_prefix)
        ][0]
        length = len(x[0])
        assert all(length == len(y) for y in x)
        return length

    def __call__(self, batch: Batch, **kwargs) -> Batch:
        n_documents = self._infer_n_docs(batch)
        batch_size = self._infer_batch_size(batch)

        # get one pair: (question and document) size: [batch_size x n_documents]
        pairs = self._get_data_pairs(batch, batch_size=batch_size)

        # potentially transform all pairs (in batch)
        pairs = self.preprocess(pairs)

        # apply self.classify element-wise (to each pair)
        results = list(map(self.classify, pairs))

        # reshape as [batch_size, n_documents] and cast as Tensor
        results = torch.tensor(results).view(batch_size, n_documents)

        # return results
        batch[self.output_key] = results
        return batch

    def _get_data_pairs(
        self, batch: Batch, batch_size: Optional[int] = None
    ) -> Iterable[Pair]:
        batch_size = batch_size or self._infer_batch_size(batch)
        for i in range(batch_size):
            a_data_i = self.get_eg(
                batch, i, filter_op=KeyWithPrefix(self.answer_prefix)
            )
            d_data_i = self.get_eg(
                batch, i, filter_op=KeyWithPrefix(self.document_prefix)
            )

            # iterate through each document
            n_docs = len(next(iter(d_data_i.values())))
            for j in range(n_docs):
                d_data_ij = {k: v[j] for k, v in d_data_i.items()}
                yield Pair(answer=a_data_i, document=d_data_ij)

    def _infer_batch_size(self, batch):
        bs = len(next(iter(batch.values())))
        assert all(bs == len(x) for x in batch.values())
        return bs


class MetaMapMatch(RelevanceClassifier):
    def __init__(self, model_name: Optional[str] = "en_core_sci_lg", **kwargs):
        super().__init__(**kwargs)

        self.model_name = model_name
        self.model = spacy.load(
            self.model_name,
            disable=[
                "tok2vec",
                "tagger",
                "parser",
                "attribute_ruler",
                "lemmatizer",
            ],
        )
        self.model.add_pipe(
            "scispacy_linker",
            config={
                "linker_name": "umls",
                "max_entities_per_mention": 3,
                "threshold": 0.95,
            },
        )
        self.linker = self.model.get_pipe("scispacy_linker")

    def classify(self, pair: Pair) -> bool:
        doc_text = pair.document["document.text"]
        answer_index = pair.answer["answer.target"]
        answer_aliases = [pair.answer["answer.text"][answer_index]]
        answer_cui = pair.answer["answer.cui"][0]
        answer_synonyms = pair.answer["answer.synonyms"]

        e_aliases = [
            alias.lower()
            for alias in self.linker.kb.cui_to_entity[answer_cui][2]
        ]

        answer_aliases = answer_aliases + answer_synonyms + e_aliases

        # re.search: Scan through string looking for a location where the regular expression pattern produces a match, and return a corresponding MatchObject instance. Return None if no position in the string matches the pattern; note that this is different from finding a zero-length match at some point in the string.
        # re.escape: Return string with all non-alphanumerics backslashed; this is useful if you want to match an arbitrary literal string that may have regular expression metacharacters in it.
        # re.IGNORECASE: Perform case-insensitive matching
        return find_one(doc_text, answer_aliases, sort_by=len)


class ScispaCyMatch(RelevanceClassifier):
    model: Optional[Language] = None
    linker: Optional[Callable[[Doc], Doc]] = None

    def __init__(
        self,
        model_name: Optional[str] = "en_core_sci_lg",
        lazy_setup: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.model_name = model_name
        if not lazy_setup:
            self._setup_models()

    def __call__(self, batch: Batch, **kwargs) -> Batch:
        """Super-charge the __call__ method to load the spaCy models
        if they are not already loaded."""
        self._setup_models()
        return super().__call__(batch, **kwargs)

    def __getstate__(self):
        """this method is called when attempting pickling"""
        state = self.__dict__.copy()
        # Serialize
        for key in ["linker", "model"]:
            if key in state:
                del state[key]
                state[key] = None

        return state

    def fingerprint(self) -> Any:
        return {
            k: self._fingerprint(v) for k, v in self.__getstate__().items()
        }

    def dill_inspect(self, reduce=True) -> Dict:
        return {**{k: dill.pickles(v) for k, v in self.__getstate__().items()}}

    def _setup_models(self):
        if self.model is None:
            self.model = self._load_spacy_model(self.model_name)
        if self.linker is None:
            self.linker = self._setup_linker(self.model)

    @staticmethod
    def _setup_linker(model: Language):
        if model is None:
            return None
        return model.get_pipe("scispacy_linker")

    @staticmethod
    def _load_spacy_model(model_name: str):
        model = spacy.load(
            model_name,
            disable=[
                "tok2vec",
                "tagger",
                "parser",
                "attribute_ruler",
                "lemmatizer",
            ],
        )
        model.add_pipe(
            "scispacy_linker",
            config={
                "linker_name": "umls",
                "max_entities_per_mention": 3,
                "threshold": 0.95,
            },
        )
        return model


    def extract_aliases(self, entity) -> Iterable[str]:
        # get the list of linked entity
        linked_entities = self.get_linked_entities(entity)

        # filter irrelevant entities based on TUIs
        # def keep_entity(ent: dict) -> bool:
        #     """
        #     keep entities that are not in the DISCARD_TUIs list.
        #     """
        #     return any(tui not in DISCARD_TUIs for tui in ent['tui'])
        filtered_entities = filter(lambda ent: any(tui not in DISCARD_TUIs for tui in ent['tui']), linked_entities)

        for linked_entity in filtered_entities:
            for alias in linked_entity['aliases']:
                yield alias.lower()

    def get_linked_entities(self, entity: Entity) -> Iterable[dict]:
        for cui in entity._.kb_ents:
            # print(cui)
            cui_str, _ = cui  # ent: (str, score)
            out = {
                "entity" : str(entity), 
                "tui" : self.linker.kb.cui_to_entity[cui_str].types, 
                "aliases" : self.linker.kb.cui_to_entity[cui_str].aliases}
            yield out

    def classify(self, pair: Pair) -> bool:

        doc_text = pair.document["document.text"]
        answer_aliases = pair.answer["answer.aliases"]
        print("Final aliases")
        print(answer_aliases)
        return find_one(doc_text, answer_aliases, sort_by=len)

    def keep_entity(ent: dict) -> bool:
            """
            keep entities that are not in the DISCARD_TUIs list.
            """
            return any(tui not in DISCARD_TUIs for tui in ent['tui'])

    @staticmethod
    def _answer_text(pair: Pair) -> str:
        answer_index = pair.answer["answer.target"]
        return pair.answer["answer.text"][answer_index]

    @staticmethod
    def _synonym_text(pair: Pair) -> str:
        return ' '.join(synonym for synonym in [pair.answer["answer.synonyms"]])

    def filter_synonyms(self, synonyms) -> Iterable[str]:
        for entity in synonyms.ents:
            # get the list of linked synonyms
            linked_synonyms = self.get_linked_entities(entity)

    def preprocess(self, pairs: Iterable[Pair]) -> Iterable[Pair]:
        """Generate the field `pair.answer["aliases"]`"""
        # An iterator can only be consumed once, generate two of them
        # casting as a list would also work
        pairs_1, pairs_2 = tee(pairs, 2)

        # extract the answer text from each Pair
        texts = map(self._answer_text, pairs_1)

        # batch processing of texts
        docs = self.model.pipe(texts)

        # join the aliases
        for pair, doc in zip_longest(pairs_2, docs):
            # the dict method get allows you to provide a default value if the key is missing
            answer_synonyms = set(pair.answer.get("answer.synonyms", []))
            answer_aliases = set.union({str(doc)}, answer_synonyms)
            for ent in doc.ents:
                e_aliases = set(self.extract_aliases(ent))
                print("extracted aliases: ", e_aliases)
                print("answer aliases: ", answer_aliases)
                answer_aliases = set.union(answer_aliases, e_aliases)

            # update the pair and return
            pair.answer["answer.aliases"] = answer_aliases
            yield pair


class ExactMatch(RelevanceClassifier):
    """Match the lower-cased answer string in the document."""

    def classify(self, pair: Pair) -> bool:
        doc_text = pair.document["document.text"]
        answer_index = pair.answer["answer.target"]
        answer_text = pair.answer["answer.text"][answer_index]

        return bool(answer_text.lower() in doc_text.lower())
