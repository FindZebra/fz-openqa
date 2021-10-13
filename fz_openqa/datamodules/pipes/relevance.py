import re
from dataclasses import dataclass
from itertools import tee
from itertools import zip_longest
from typing import Any
from typing import Dict
from typing import Iterable
from typing import Optional

import numpy as np
import rich
import spacy
import torch
from scispacy.abbreviation import AbbreviationDetector  # type: ignore
from scispacy.linking import EntityLinker  # type: ignore
from scispacy.linking_utils import Entity

from .static import DISCARD_TUIs
from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.utils.datastruct import Batch

np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


@dataclass
class Pair:
    document: Dict[str, Any]
    answer: Dict[str, Any]


class RelevanceClassifier(Pipe):
    def __init__(
        self,
        answer_prefix: str = "answer.",
        document_prefix: str = "document.",
        output_key: str = "document.is_positive",
        output_count_key: str = "document.positive_count",
    ):
        self.output_key = output_key
        self.answer_prefix = answer_prefix
        self.document_prefix = document_prefix
        self.output_count_key = output_count_key

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
        batch[self.output_count_key] = results.float().sum(-1).long()
        return batch

    def _get_data_pairs(
        self, batch: Batch, batch_size: Optional[int] = None
    ) -> Iterable[Pair]:
        batch_size = batch_size or self._infer_batch_size(batch)
        for i in range(batch_size):
            a_data_i = {
                k: v[i] for k, v in batch.items() if self.answer_prefix in k
            }
            d_data_i = {
                k: v[i] for k, v in batch.items() if self.document_prefix in k
            }

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
        super().__init__()

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

    def extract_aliases(self, cui) -> Iterable[str]:
        for alias in self.linker.kb.cui_to_entity[cui][2]:
            yield alias.lower()

    def classify(self, pair: Pair) -> bool:

        doc_text = pair.document["document.text"]
        answer_aliases = pair.answer["answer.aliases"]

        # re.search: Scan through string looking for a location where the regular expression pattern produces a match, and return a corresponding MatchObject instance. Return None if no position in the string matches the pattern; note that this is different from finding a zero-length match at some point in the string.
        # re.escape: Return string with all non-alphanumerics backslashed; this is useful if you want to match an arbitrary literal string that may have regular expression metacharacters in it.
        # re.IGNORECASE: Perform case-insensitive matching
        return bool(
            re.search(
                re.compile(
                    "|".join(re.escape(x) for x in answer_aliases),
                    re.IGNORECASE,
                ),
                doc_text,
            )
        )
    
    @staticmethod
    def _answer_text(pair: Pair) -> str:
        answer_index = pair.answer["answer.target"]
        return pair.answer["answer.text"][answer_index]

    def preprocess(self, pairs: Iterable[Pair]) -> Iterable[Pair]:
        """Generate the field `pair.answer["aliases"]`"""
        # An iterator can only be consumed once, generate two of them
        # casting as a list would also work
        pairs_1, pairs_2 = tee(pairs, 2)

        # extract the answer text from each Pair
        texts = map(self._answer_text, pairs_1)

        # join the aliases
        for pair, text in zip_longest(pairs_2, texts):
            answer_cuis = pair.answer.get("answer.cui", None)
            answer_synonyms = set(pair.answer.get("answer.synonyms", []))
            answer_aliases = set.union({str(text)}, answer_synonyms)
            if len(answer_cuis)>0:
                e_aliases = set(self.extract_aliases(answer_cuis[0]))
                answer_aliases = set.union(answer_aliases, e_aliases)

            # update the pair and return
            pair.answer["answer.aliases"] = answer_aliases
            yield pair

class ScispaCyMatch(RelevanceClassifier):
    def __init__(self, model_name: Optional[str] = "en_core_sci_lg", **kwargs):
        super().__init__()

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

    def extract_aliases(self, entity) -> Iterable[str]:
        # get the list of linked entity
        linked_entities = self.get_linked_entities(entity)

        # filter irrelevant entities based on TUIs
        def keep_entity(ent: Entity) -> bool:
            """
            keep entities that are not in the DISCARD_TUIs list.
            """

            return any(tui not in DISCARD_TUIs for tui in ent.types)

        linked_entities = filter(lambda ent: keep_entity, linked_entities)

        # return aliases
        for linked_entity in linked_entities:
            for alias in linked_entity.aliases:
                yield alias.lower()

    def get_linked_entities(self, entity: Entity) -> Iterable[Entity]:
        for cui in entity._.kb_ents:
            print(cui)
            cui_str, _ = cui  # ent: (str, score)
            yield self.linker.kb.cui_to_entity[cui_str]

    def classify(self, pair: Pair) -> bool:

        doc_text = pair.document["document.text"]
        answer_aliases = pair.answer["answer.aliases"]

        # re.search: Scan through string looking for a location where the regular expression pattern produces a match, and return a corresponding MatchObject instance. Return None if no position in the string matches the pattern; note that this is different from finding a zero-length match at some point in the string.
        # re.escape: Return string with all non-alphanumerics backslashed; this is useful if you want to match an arbitrary literal string that may have regular expression metacharacters in it.
        # re.IGNORECASE: Perform case-insensitive matching
        return bool(
            re.search(
                re.compile(
                    "|".join(re.escape(x) for x in answer_aliases),
                    re.IGNORECASE,
                ),
                doc_text,
            )
        )

    @staticmethod
    def _answer_text(pair: Pair) -> str:
        answer_index = pair.answer["answer.target"]
        return pair.answer["answer.text"][answer_index]

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

                # todo: why sorting here? doesn't look necessary  
                # answer_aliases = sorted(set.union(answer_aliases, e_aliases), key=len)
                answer_aliases = set.union(answer_aliases, e_aliases)
            # todo: investigate the aliases, some are too general
            rich.print(
                f">> aliases={answer_aliases}, "
                f"count={len(answer_aliases)}, "
                f"doc_ents={doc.ents}"
            )

            # update the pair and return
            pair.answer["answer.aliases"] = answer_aliases
            yield pair


class ExactMatch(RelevanceClassifier):
    """Match the lower-cased answer string in the document."""

    def classify(self, pair: Pair) -> bool:
        doc_text = pair.document["document.text"]
        answer_index = pair.answer["answer.target"]
        answer_text = pair.answer["answer.text"][answer_index]

        return bool(answer_text in doc_text)
