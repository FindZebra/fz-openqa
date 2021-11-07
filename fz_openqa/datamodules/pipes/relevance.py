import re
from dataclasses import dataclass
from functools import partial
from itertools import chain
from itertools import tee
from itertools import zip_longest
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

import dill
import numpy as np
import spacy
from scispacy.abbreviation import AbbreviationDetector  # type: ignore
from scispacy.linking import EntityLinker  # type: ignore
from scispacy.linking_utils import Entity
from spacy import Language
from spacy.tokens import Doc
from spacy.tokens.span import Span

from .nesting import nested_list
from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.datamodules.pipes.control.filter_keys import KeyWithPrefix
from fz_openqa.datamodules.pipes.utils.static import DISCARD_TUIs
from fz_openqa.datamodules.pipes.utils.static import STOP_WORDS
from fz_openqa.utils.datastruct import Batch

np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


@dataclass
class LinkedEntity:
    entity: str
    tuis: List[str]
    aliases: List[str]


@dataclass
class Pair:
    document: Dict[str, Any]
    answer: Dict[str, Any]


def find_one(text: str, queries: Sequence[Any], sort_by: Optional[Callable] = None) -> bool:
    """check if one of the queries is in the input text"""
    assert isinstance(text, str)
    if len(queries) == 0:
        return False
    if len(text) == 0:
        return False

    if sort_by is not None:
        queries = sorted(queries, key=sort_by)

    # re.search: Scan through string looking for a location where
    # the regular expression pattern produces a match, and return a
    # corresponding MatchObject instance. Return None if no position
    # in the string matches the pattern; note that this is different
    # from finding a zero-length match at some point in the string.
    # re.escape: Return string with all non-alphanumerics backslashed;
    # this is useful if you want to match an arbitrary literal string
    # that may have regular expression metacharacters in it.
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


def find_all(text: str, queries: Sequence[Any], lower_case_queries: bool = True) -> List:
    """Find all matching queries in the document.
    There are one returned item per match in the document."""
    assert isinstance(text, str)
    if len(queries) == 0:
        return []
    if len(text) == 0:
        return []

    if lower_case_queries:
        queries = {q.lower() for q in queries}

    # re.search: Scan through string looking for a location where
    # the regular expression pattern produces a match, and return a
    # corresponding MatchObject instance. Return None if no position
    # in the string matches the pattern; note that this is different
    # from finding a zero-length match at some point in the string.
    # re.escape: Return string with all non-alphanumerics backslashed;
    # this is useful if you want to match an arbitrary literal string
    # that may have regular expression metacharacters in it.
    # re.IGNORECASE: Perform case-insensitive matching
    return re.findall(
        re.compile(
            "|".join(re.escape(x) for x in queries),
            re.IGNORECASE,
        ),
        text,
    )


class RelevanceClassifier(Pipe):
    def __init__(
        self,
        answer_prefix: str = "answer.",
        document_prefix: str = "document.",
        output_key: str = "document.match_score",
        interpretable: bool = False,
        interpretation_key: str = "document.match_on",
        id="relevance-classifier",
        **kwargs,
    ):
        super(RelevanceClassifier, self).__init__(id=id)
        self.output_key = output_key
        self.answer_prefix = answer_prefix
        self.document_prefix = document_prefix
        self.interpretable = interpretable
        self.interpretation_key = interpretation_key

    def output_keys(self, input_keys: List[str]) -> List[str]:
        input_keys = [self.output_key]
        if self.interpretable:
            input_keys += [self.interpretation_key]
        return input_keys

    @staticmethod
    def _get_matches(pair: Pair) -> List[str]:
        """Return the list of matches given the pair of data.
        Needs to be implemented for each sub-class."""
        raise NotImplementedError

    def classify(self, pair: Pair) -> int:
        matches = self._get_matches(pair)
        return len(matches)

    def classify_and_interpret(self, pair: Pair) -> Tuple[int, List[str]]:
        matches = self._get_matches(pair)
        return (len(matches), matches)

    def preprocess(self, pairs: Iterable[Pair]) -> Iterable[Pair]:
        """Preprocessing allows transforming all the pairs,
        potentially in batch mode."""
        return pairs

    def _infer_n_docs(self, batch: Batch) -> int:
        x = [v for k, v in batch.items() if str(k).startswith(self.document_prefix)][0]
        length = len(x[0])
        assert all(length == len(y) for y in x)
        return length

    def __call__(self, batch: Batch, **kwargs) -> Batch:
        output = {}
        n_documents = self._infer_n_docs(batch)
        batch_size = self._infer_batch_size(batch)

        # get one pair: (question and document) size: [batch_size x n_documents]
        pairs = self._get_data_pairs(batch, batch_size=batch_size)

        # potentially transform all pairs (in batch)
        pairs = self.preprocess(pairs)

        # apply self.classify element-wise (to each pair)
        if self.interpretable:
            all_results = zip(*map(self.classify_and_interpret, pairs))
            results, interpretations = map(list, all_results)
            output[self.interpretation_key] = nested_list(interpretations, stride=n_documents)
        else:
            results = list(map(self.classify, pairs))

        # reshape as [batch_size, n_documents] and cast as Tensor
        output[self.output_key] = nested_list(results, stride=n_documents)
        return output

    def _get_data_pairs(self, batch: Batch, batch_size: Optional[int] = None) -> Iterable[Pair]:
        batch_size = batch_size or self._infer_batch_size(batch)
        for i in range(batch_size):
            a_data_i = self.get_eg(batch, i, filter_op=KeyWithPrefix(self.answer_prefix))
            d_data_i = self.get_eg(batch, i, filter_op=KeyWithPrefix(self.document_prefix))

            # iterate through each document
            n_docs = len(next(iter(d_data_i.values())))
            for j in range(n_docs):
                d_data_ij = {k: v[j] for k, v in d_data_i.items()}
                yield Pair(answer=a_data_i, document=d_data_ij)

    def _infer_batch_size(self, batch):
        bs = len(next(iter(batch.values())))
        assert all(bs == len(x) for x in batch.values())
        return bs


class ExactMatch(RelevanceClassifier):
    """Match the lower-cased answer string in the document."""

    @staticmethod
    def _get_matches(pair: Pair) -> List[str]:
        doc_text = pair.document["document.text"]
        answer_index = pair.answer["answer.target"]
        answer_text = pair.answer["answer.text"][answer_index]
        answer_text = " ".join(
            [word for word in answer_text.split() if word.lower() not in STOP_WORDS]
        )

        return find_all(doc_text, [answer_text])

    # def preprocess(self, pairs: Iterable[Pair]) -> Iterable[Pair]:
    #     """Generate the field `pair.answer["aliases"]`"""
    #     # An iterator can only be consumed once, generate two of them
    #     # casting as a list would also work
    #     pairs_1, pairs_2 = tee(pairs, 2)


class AliasBasedMatch(RelevanceClassifier):
    model: Optional[Language] = None
    linker: Optional[Callable] = None

    def __init__(
        self,
        filter_tui: Optional[bool] = True,
        filter_acronyms: Optional[bool] = True,
        model_name: Optional[str] = "en_core_sci_lg",
        linker_name: str = "umls",
        lazy_setup: bool = True,
        spacy_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filter_tui: filter_tui
        self.filter_acronyms = filter_acronyms
        self.model_name = model_name
        self.linker_name = linker_name
        self.spacy_kwargs = spacy_kwargs or {"batch_size": 100, "n_process": 1}
        if not lazy_setup:
            self._setup_models()

    @staticmethod
    def _get_matches(pair: Pair) -> List[str]:
        doc_text = pair.document["document.text"]
        answer_aliases = pair.answer["answer.aliases"]
        return find_all(doc_text, answer_aliases)

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
        return {k: self._fingerprint(v) for k, v in self.__getstate__().items()}

    def dill_inspect(self, reduce=True) -> Dict:
        return {k: dill.pickles(v) for k, v in self.__getstate__().items()}

    def _setup_models(self):
        if self.model is None:
            self.model = self._load_spacy_model(self.model_name, self.linker_name)
        if self.linker is None:
            self.linker = self._setup_linker(self.model)

    @staticmethod
    def _setup_linker(model: Language):
        if model is None:
            return None
        return model.get_pipe("scispacy_linker")

    @staticmethod
    def _load_spacy_model(model_name: str, linker_name: str = "umls"):
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
                "threshold": 0.65,
                "linker_name": linker_name,
            },
        )
        return model

    def get_linked_entities(self, entity: Entity) -> Iterable[LinkedEntity]:
        for cui in entity._.kb_ents:
            cui_str, _ = cui  # ent: (str, score)
            tuis = self.linker.kb.cui_to_entity[cui_str].types
            aliases = self.linker.kb.cui_to_entity[cui_str].aliases
            yield LinkedEntity(entity=str(entity), tuis=tuis, aliases=aliases)

    @staticmethod
    def _extract_answer_text(pair: Pair) -> str:
        answer_index = pair.answer["answer.target"]
        return pair.answer["answer.text"][answer_index]

    @staticmethod
    def _extract_synonym_text(pair: Pair) -> str:
        return ",".join([synonym for synonym in pair.answer.get("answer.synonyms", [])])

    def detect_acronym(self, alias: str) -> bool:
        """
        returns true if accronym is found in string
            example: "AbIA AoP U.S.A. USA"
        """
        regex_pattern = r"\b[A-Z][a-zA-Z\.]*[A-Z]\b\.?"
        return re.match(regex_pattern, alias)

    @staticmethod
    def _check_entity_tuis(ent: LinkedEntity, *, discard_list: List[str]) -> bool:
        return any(tui not in discard_list for tui in ent.tuis)

    def extract_and_filters_entities(self, doc: Doc) -> Iterable[str]:
        for entity in doc.ents:
            linked_entities = self.get_linked_entities(entity)

            # if self.filter_tui:
            # filter irrelevant entities based on TUIs
            _filter = partial(self._check_entity_tuis, discard_list=DISCARD_TUIs)
            linked_entities = filter(_filter, linked_entities)

            for linked_entity in linked_entities:
                if not self.filter_acronyms:
                    yield linked_entity.entity.lower()
                elif self.detect_acronym(linked_entity.entity):
                    pass
                else:
                    yield linked_entity.entity.lower()

    def extract_aliases(self, linked_entities: Iterable[LinkedEntity]) -> Iterable[str]:
        # get the TUIs of linked entities to filter irrelevant ones
        # if self.filter_tuis:
        # filter irrelevant entities based on TUIs
        _filter = partial(self._check_entity_tuis, discard_list=DISCARD_TUIs)
        linked_entities = filter(_filter, linked_entities)

        for linked_entity in linked_entities:
            for alias in linked_entity.aliases:
                if not self.filter_acronyms:
                    yield alias.lower()
                elif self.detect_acronym(alias):
                    pass
                else:
                    yield alias.lower()


class MetaMapMatch(AliasBasedMatch):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def preprocess(self, pairs: Iterable[Pair]) -> Iterable[Pair]:
        """Generate the field `pair.answer["aliases"]`"""
        pairs = list(pairs)

        # extract the answer and synonym texts from each Pair
        answer_texts = map(self._extract_answer_text, pairs)
        synonym_texts = map(self._extract_synonym_text, pairs)

        # batch processing of texts
        synonym_docs: List[Doc] = self.model.pipe(synonym_texts, **self.spacy_kwargs)

        # join the aliases
        for pair, answer, synonym_doc in zip_longest(pairs, answer_texts, synonym_docs):
            answer_cuis = pair.answer.get("answer.cui", [])
            filtered_synonyms = self.extract_and_filters_entities(synonym_doc)
            answer_aliases = set(filtered_synonyms)
            if len(answer_cuis) > 0:
                for cui in answer_cuis:
                    linked_entities = self.linker.kb.cui_to_entity[cui]
                    e_aliases = set(self.extract_aliases(linked_entities))
                    answer_aliases = set.union(answer_aliases, e_aliases)

            # remove stopwords
            answer_string = " ".join(
                [word for word in answer.split() if word.lower() not in STOP_WORDS]
            )

            answer_aliases = [answer_string] + sorted(answer_aliases, key=len)
            # update the pair and return
            pair.answer["answer.aliases"] = list(answer_aliases)
            yield pair


class ScispaCyMatch(AliasBasedMatch):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def preprocess(self, pairs: Iterable[Pair]) -> Iterable[Pair]:
        """Generate the field `pair.answer["aliases"]`"""
        pairs = list(pairs)
        n = len(pairs)

        # extract the answer and synonyms texts from each Pair
        answer_texts = map(self._extract_answer_text, pairs)
        synonym_texts = map(self._extract_synonym_text, pairs)

        # batch processing of texts
        docs = list(self.model.pipe(chain(answer_texts, synonym_texts), **self.spacy_kwargs))
        answer_docs, synonym_docs = docs[:n], docs[n:]

        # join the aliases
        for pair, answer_doc, synonym_doc in zip_longest(pairs, answer_docs, synonym_docs):
            answer_doc.ents = [Span(answer_doc, 0, len(answer_doc), label="Entity")]
            linked_entities = self.get_linked_entities(answer_doc.ents[0])
            e_aliases = set(self.extract_aliases(linked_entities))

            # remove stopwords
            answer_string = " ".join(
                [word for word in str(answer_doc).split() if word.lower() not in STOP_WORDS]
            )

            answer_aliases = [answer_string] + sorted(e_aliases, key=len)

            # update the pair and return
            pair.answer["answer.aliases"] = list(answer_aliases)
            yield pair
