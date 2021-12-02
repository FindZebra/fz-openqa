import itertools
import re
from dataclasses import dataclass
from functools import partial
from itertools import zip_longest
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np
import rich
import spacy
from scispacy.linking import EntityLinker  # type: ignore
from scispacy.linking_utils import Entity
from spacy.language import Language
from spacy.tokens.span import Span

from ...utils.functional import infer_batch_size
from .base import Pipe
from fz_openqa.datamodules.pipes.control.condition import HasPrefix
from fz_openqa.datamodules.pipes.utils.spacy_pipe_functions import merge_consecutive_entities  # type: ignore  # noqa: E501
from fz_openqa.datamodules.pipes.utils.static import DISCARD_TUIs
from fz_openqa.utils.datastruct import Batch

np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


@dataclass
class LinkedEntity:
    entity: str
    name: str
    tuis: List[str]
    aliases: List[str]


@dataclass
class Pair:
    document: Dict[str, Any]
    answer: Dict[str, Any]


def find_one(text: str, queries: Sequence[Any]) -> bool:
    """check if one of the queries is in the input text"""
    assert isinstance(text, str)
    if len(queries) == 0:
        return False
    if len(text) == 0:
        return False

    return bool(
        re.findall(
            re.compile(
                "(?=(" + "|".join(map(re.escape, queries)) + "))",
                re.IGNORECASE,
            ),
            text,
        )
    )


def find_all(text: str, queries: Sequence[List], lower_case_queries: bool = True) -> List:
    """Find all matching queries in the document.
    There are one returned item per match in the document."""
    assert isinstance(text, str), f"The input must be a string. Found {type(text)}"
    if len(queries) == 0:
        return []
    if len(text) == 0:
        return []
    if lower_case_queries:
        queries = {q.lower() for q in queries}

    return re.findall(
        re.compile(
            "(?=(" + "|".join(map(re.escape, queries)) + "))",
            re.IGNORECASE,
        ),
        text,
    )


class RelevanceClassifier(Pipe):
    """
    Classify if a given document is relevant to the question.
    Returns `document.match_score`, which indicates the relevance of the
    document (higher ==> more relevant).
    If `interpretable=True`, the matched tokens in the given document (`document.match_on`)
    are also returned.
    """

    def __init__(
        self,
        answer_field: str = "answer",
        document_field: str = "document",
        output_key: str = "document.match_score",
        interpretable: bool = False,
        interpretation_key: str = "document.match_on",
        id="relevance-classifier",
        **kwargs,
    ):
        super(RelevanceClassifier, self).__init__(id=id)
        self.output_key = output_key
        self.answer_field = answer_field
        self.document_field = document_field
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
        x = [v for k, v in batch.items() if str(k).startswith(self.document_field)][0]
        length = len(x[0])
        assert all(length == len(y) for y in x)
        return length

    def _call_batch(self, batch: Batch, **kwargs) -> Batch:
        output = {}
        # get pairs of data: Pair(answer, document)
        pairs = self._get_data_pairs(batch)

        # potentially transform all pairs (in batch)
        pairs = self.preprocess(pairs)

        # apply self.classify element-wise (to each pair)
        if self.interpretable:
            all_results = zip(*map(self.classify_and_interpret, pairs))
            results, interpretations = map(list, all_results)
            output[self.interpretation_key] = list(interpretations)
        else:
            results = list(map(self.classify, pairs))

        # reshape as [batch_size, n_documents] and cast as Tensor
        output[self.output_key] = list(results)

        return output

    def _get_data_pairs(self, batch: Batch, batch_size: Optional[int] = None) -> Iterable[Pair]:
        batch_size = batch_size or infer_batch_size(batch)
        for i in range(batch_size):
            a_data_i = self.get_eg(batch, i, filter_op=HasPrefix(self.answer_field))
            d_data_i = self.get_eg(batch, i, filter_op=HasPrefix(self.document_field))
            yield Pair(answer=a_data_i, document=d_data_i)


class ExactMatch(RelevanceClassifier):
    """Match the lower-cased answer string in the document."""

    def _get_matches(self, pair: Pair) -> List[str]:
        doc_text = pair.document[f"{self.document_field}.text"]
        answer_text = pair.answer[f"{self.answer_field}.text"]
        return find_all(doc_text, [answer_text])


class AliasBasedMatch(RelevanceClassifier):
    """Relevance Classifier based on Entity aliases"""

    model: Optional[Language] = None
    linker: Optional[Callable] = None

    def __init__(
        self,
        filter_tui: Optional[bool] = True,
        filter_acronyms: Optional[bool] = True,
        model_name: Optional[str] = "en_core_sci_lg",
        linker_name: str = "umls",
        threshold: float = 0.7,
        lazy_setup: bool = True,
        spacy_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        filter_tui
            Filter aliases according to DISCARD_TUIs list
        filter_acronyms
            Filter aliases according to regex pattern catching acronyms
        model_name
            String defining what ScispaCy model to use
            see: https://github.com/allenai/scispacy#available-models
        linker_name
            String defining what knowledge base to use as Linker
            see: https://github.com/allenai/scispacy#entitylinker
        threshold
            Threshold that a mention candidate must reach to be added
            to the mention in the Doc as a mention candidate.
        lazy_setup
            If True, the model and linker will be loaded only when needed.
        spacy_kwargs
            Keyword arguments to pass to the ScispaCy model.
        """
        super().__init__(**kwargs)
        self.filter_tui = filter_tui
        self.filter_acronyms = filter_acronyms
        self.model_name = model_name
        self.threshold = threshold
        self.linker_name = linker_name
        self.spacy_kwargs = spacy_kwargs or {"batch_size": 100, "n_process": 1}
        if not lazy_setup:
            self._setup_models()

    def _get_matches(self, pair: Pair) -> List[str]:
        doc_text = pair.document[f"{self.document_field}.text"]
        answer_aliases = pair.answer[f"{self.answer_field}.aliases"]
        return find_all(doc_text, answer_aliases)

    def _call_batch(self, batch: Batch, **kwargs) -> Batch:
        """Super-charge the __call__ method to load the spaCy models
        if they are not already loaded."""
        self._setup_models()
        return super()._call_batch(batch, **kwargs)

    def __getstate__(self):
        """this method is called when attempting pickling"""
        state = self.__dict__.copy()
        # Serialize
        for key in ["linker", "model"]:
            if key in state:
                del state[key]
                state[key] = None

        return state

    def _setup_models(self):
        if self.model is None:
            self.model = self._load_spacy_model(self.model_name, self.linker_name, self.threshold)
        if self.linker is None:
            self.linker = self._setup_linker(self.model)

    @staticmethod
    def _setup_linker(model: Language):
        if model is None:
            return None
        return model.get_pipe("scispacy_linker")

    @staticmethod
    def _load_spacy_model(model_name: str, linker_name: str = "umls", threshold: float = 0.7):
        """When you call a spaCy model on a text, spaCy first tokenizes the text to produce a Doc object.
        Doc is then processed in several different steps – the processing pipeline."""
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
            "merge_consecutive_entities",
        )
        model.add_pipe(
            "scispacy_linker",
            config={
                "threshold": threshold,
                "linker_name": linker_name,
                "max_entities_per_mention": 3,
            },
        )
        return model

    @staticmethod
    def _filter_entities(entity: Tuple, name: str) -> Entity:
        _, score = entity  # ent: (str, score)
        if name.count(" ") > 1:
            return True
        elif score == 1.0:
            return True
        else:
            False

    def get_linked_entities(self, entity: Span) -> Iterable[LinkedEntity]:
        """ Extracts the linked entities by querying the Doc entity against the knowledge base"""
        entities = filter(
            lambda ent: self._filter_entities(entity=ent, name=entity.text), entity._.kb_ents
        )
        for cui in entities:
            cui_str, _ = cui  # ent: (str, score)
            tuis = self.linker.kb.cui_to_entity[cui_str].types
            name = self.linker.kb.cui_to_entity[cui_str].canonical_name
            aliases = self.linker.kb.cui_to_entity[cui_str].aliases
            yield LinkedEntity(entity=str(entity), name=name, tuis=tuis, aliases=aliases)

    def _extract_answer_text(self, pair: Pair) -> str:
        return pair.answer[f"{self.answer_field}.text"]

    @staticmethod
    def detect_acronym(alias: str) -> bool:
        """Regex pattern to detect acronym.

        Parameters
        ----------
        alias : str
            The string representing an alias

        Returns
        ------
        bool
            True if accronym is found in string

        Examples
        --------
        >>> print(AliasBasedMatch.detect_acronym("AbIA|AoP|U.S.A.|USA")
        True
        """
        regex_pattern = r"\b[A-Z][a-zA-Z\.]*[A-Z]\b\.?"
        return re.match(regex_pattern, alias)

    @staticmethod
    def _check_entity_tuis(ent: LinkedEntity, *, discard_list: List[str]) -> bool:
        return any(tui not in discard_list for tui in ent.tuis)

    @staticmethod
    def _filter_aliases(aliases: List[str], *, name: str) -> LinkedEntity:
        return any(alias.lower() not in name.lower() for alias in aliases)

    def extract_aliases(self, linked_entities: Iterable[LinkedEntity]) -> Iterable[str]:
        """ Extract aliases of the linked entities"""
        # get the TUIs of linked entities to filter irrelevant ones
        # filter irrelevant entities based on TUIs
        if self.filter_tui:
            _filter = partial(self._check_entity_tuis, discard_list=DISCARD_TUIs)
            linked_entities = filter(_filter, linked_entities)

        for linked_entity in linked_entities:
            # filter alias if aliases part of entity name
            _filter = partial(self._filter_aliases, name=linked_entity.name)
            aliases = list(filter(_filter, linked_entity.aliases))
            aliases.insert(0, linked_entity.name)
            for alias in aliases:
                if len(alias) <= 3:
                    pass
                elif not self.filter_acronyms:
                    yield alias.lower()
                elif self.detect_acronym(alias):
                    pass
                else:
                    yield alias.lower()


class ScispaCyMatch(AliasBasedMatch):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def preprocess(self, pairs: Iterable[Pair]) -> Iterable[Pair]:
        """Generate the field `pair.answer["aliases"]`"""
        pairs = list(pairs)
        # extract the answer and synonyms texts from each Pair
        answer_texts = map(self._extract_answer_text, pairs)
        # batch processing of texts
        docs = list(self.model.pipe(answer_texts, **self.spacy_kwargs))

        # join the aliases
        for pair, answer_doc in zip_longest(pairs, docs):
            answer_str = answer_doc.text
            e_aliases = set()
            for ent in answer_doc.ents:
                linked_entities = self.get_linked_entities(ent)
                e_aliases = set.union(set(self.extract_aliases(linked_entities)), e_aliases)

            answer_aliases = [answer_str] + list(e_aliases)
            # update the pair and return
            pair.answer[f"{self.answer_field}.aliases"] = answer_aliases
            yield pair
