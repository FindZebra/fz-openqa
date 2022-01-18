import itertools
import re
import warnings
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
from dataclasses import dataclass
from scispacy.linking import EntityLinker  # type: ignore
from scispacy.linking_utils import Entity
from spacy.language import Language
from spacy.tokens.doc import Doc
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
    tuis: str
    name: str
    aliases: List[str]


@dataclass
class Pair:
    document: Dict[str, Any]
    answer: Dict[str, Any]


def find_one(text: str, query: str) -> List:
    """check if one of the queries is in the input text"""
    assert isinstance(text, str)
    assert isinstance(query, str)

    return re.findall(
        re.compile(
            "(?=(" + re.escape(query) + "))",
            re.IGNORECASE,
        ),
        text,
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
        queries = [q.lower() for q in queries]

    return re.findall(
        re.compile(
            "(?=(" + "\\b|\\b".join(map(re.escape, queries)) + r"\b))",
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
        if doc_text is None or answer_text is None:
            raise ValueError(f"Missing text for document or answer: {pair}")
        return find_one(doc_text, answer_text)


class AliasBasedMatch(RelevanceClassifier):
    """Relevance Classifier based on Entity aliases"""

    model: Optional[Language] = None
    linker: Optional[Callable] = None

    def __init__(
        self,
        filter_tui: Optional[bool] = True,
        filter_acronyms: Optional[bool] = True,
        model_name: Optional[str] = "en_core_sci_scibert",
        linker_name: str = "mesh",
        threshold: float = 0.7,
        n_entities: int = 3,
        version: str = "v2",
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
        if linker_name == "mesh":
            self.filter_tui = False
        else:
            self.filter_tui = filter_tui
        self.filter_acronyms = filter_acronyms
        self.model_name = model_name
        self.threshold = threshold
        self.linker_name = linker_name
        self.n_entities = n_entities
        self.version = version
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
            self.model = self._load_spacy_model(
                self.model_name, self.linker_name, self.threshold, self.n_entities, self.version
            )
        if self.linker is None:
            self.linker = self._setup_linker(self.model)

    @staticmethod
    def _setup_linker(model: Language):
        if model is None:
            return None
        return model.get_pipe("scispacy_linker")

    @staticmethod
    def _load_spacy_model(
        model_name: str,
        linker_name: str = "umls",
        threshold: float = 0.5,
        n_entities: int = 3,
        version: str = "v2",
    ):
        """When you call a spaCy model on a text, spaCy first tokenizes the text to produce a Doc object.
        Doc is then processed in several different steps – the processing pipeline."""
        # initialize gpu usage if possible
        # spacy.prefer_gpu()

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
        # use a custom tokenizer to speed up processing time
        # model.tokenizer = WhitespaceTokenizer(model.vocab)
        if version == "v1":
            model.add_pipe("merge_entities")
        elif version == "v2":
            model.add_pipe("merge_consecutive_entities")

        model.add_pipe(
            "scispacy_linker",
            config={
                "threshold": threshold,
                "linker_name": linker_name,
                "max_entities_per_mention": n_entities,
            },
        )
        return model

    def get_linked_entities(self, entity: Span) -> Iterable[LinkedEntity]:
        """ Extracts the linked entities by querying the Doc entity against the knowledge base"""
        for cui in entity._.kb_ents:
            cui_str, _ = cui  # ent: (str, score)
            tuis = self.linker.kb.cui_to_entity[cui_str].types
            name = self.linker.kb.cui_to_entity[cui_str].canonical_name
            aliases = self.linker.kb.cui_to_entity[cui_str].aliases

            yield LinkedEntity(entity=str(entity), tuis=tuis, name=name, aliases=aliases)

    def _extract_answer_text(self, pair: Pair) -> str:
        return pair.answer[f"{self.answer_field}.text"]

    @staticmethod
    def _check_entity_tuis(ent: LinkedEntity, *, discard_list: List[str]) -> bool:
        return any(tui not in discard_list for tui in ent.tuis)

    @staticmethod
    def _filter_aliases(alias: str, *, name: str) -> str:
        patterns = name.lower().split()
        return any(alias.lower() not in pattern for pattern in patterns)

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
                if 4 <= len(alias) <= len(linked_entity.name) * 1.5:
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


# import re
# queries = ['amox', 'amoxelline']
# text = 'this is drug called amoxelline. which can cure people'
# re.findall(
#         re.compile(
#             "(?=(\b" + '\\b|\\b'.join(queries) + r"\b))",
#             re.IGNORECASE,
#         ),
#         text,
#     )
# re.findall(
#     re.compile(
#         r"(?=(\b" + '\\b|\\b'.join(map(re.escape, queries) + r"\b))",
#                                    re.IGNORECASE,
#                                    ),
#         text,
#     )
#
# re.findall(
#         re.compile(
#             "(?=(" + '\\b|\\b'.join(queries) + r"\b))",
#             re.IGNORECASE,
#         ),
#         text,
#     )
