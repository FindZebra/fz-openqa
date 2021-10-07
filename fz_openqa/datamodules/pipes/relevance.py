import re
from typing import Any
from typing import Dict
from typing import Iterable
from typing import Optional

import numpy as np
import rich
import spacy
import torch
from scispacy.linking_utils import Entity
from scispacy.abbreviation import AbbreviationDetector  # type: ignore
from scispacy.linking import EntityLinker  # type: ignore

from .static import DISCARD_TUIs
from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.utils.datastruct import Batch
from fz_openqa.datamodules.pipes.text_ops import TextCleaner


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

    def classify(self, question: Any, document: Any) -> bool:
        raise NotImplementedError

    def __call__(self, batch: Batch, **kwargs) -> Batch:
        results = []
        batch_size = len(next(iter(batch.values())))
        for i in range(batch_size):
            q_data_i = {
                k: v[i] for k, v in batch.items() if self.answer_prefix in k
            }
            d_data_i = {
                k: v[i] for k, v in batch.items() if self.document_prefix in k
            }

            # iterate through each document
            results_i = []
            n_docs = len(next(iter(d_data_i)))
            for j in range(n_docs):
                d_data_ij = {k: v[j] for k, v in d_data_i.items()}
                results_i += [self.classify(q_data_i, d_data_ij)]
            results += [results_i]

        results = torch.tensor(results)
        batch[self.output_key] = results
        batch[self.output_count_key] = results.float().sum(-1).long()
        return batch


class MetaMapMatch(RelevanceClassifier):
    def __init__(self, model_name: Optional[str] = "en_core_sci_lg", **kwargs):
        super().__init__()
        from scispacy.linking import EntityLinker  # type: ignore

        self.model_name = model_name
        self.model = spacy.load(self.model_name)
        self.model.add_pipe(
            "scispacy_linker",
            config={
                "linker_name" : "umls",
                "max_entities_per_mention" : 3,
                "threshold" : 0.95
            },
        )
        self.linker = self.model.get_pipe("scispacy_linker")
    
    def get_linked_entities(self, cui: str) -> Iterable[Entity]:
        yield self.linker.kb.cui_to_entity[cui]

    def classify(
        self, answer: Dict[str, Any], document: Dict[str, Any]
    ) -> bool:
        doc_text = document["document.text"]
        answer_index = answer["answer.target"]
        answer_aliases = [answer["answer.text"][answer_index]]
        answer_cui = answer["answer.cui"][0]
        answer_synonyms = answer["answer.synonyms"]

        e_aliases = [alias.lower() for alias in self.linker.kb.cui_to_entity[answer_cui][2]]

        answer_aliases = set.union(answer_aliases, answer_synonyms, e_aliases)
        
        return bool(
            re.search(
                re.compile('|'.join(re.escape(x) for x in answer_aliases), re.IGNORECASE)
                , doc_text)
        )


class SciSpacyMatch(RelevanceClassifier):
    def __init__(self, model_name: Optional[str] = "en_core_sci_lg", **kwargs):
        super().__init__()

        self.model_name = model_name
        self.model = spacy.load(self.model_name)
        self.model.add_pipe(
            "scispacy_linker",
            config={
                "linker_name" : "umls",
                "max_entities_per_mention" : 3,
                "threshold" : 0.95
            },
        )
        self.linker = self.model.get_pipe("scispacy_linker")
    
    def extract_aliases(self, entity) -> Iterable[str]:
        # get the list of linked entity
        linked_entities = self.get_linked_entities(entity)

        # filter irrelevant entities based on TUIs
        def keep_entity(ent: Entity) -> bool:
            """
            keep those entities not in the DISCARD_TUIs list.
            """
            
            return bool(tui not in DISCARD_TUIs for tui in ent.types)

        linked_entities = filter(lambda ent: keep_entity, linked_entities)

        # return aliases
        for linked_entity in linked_entities:
            for alias in linked_entity.aliases:
                yield alias.lower()

    def get_linked_entities(self, entity: Entity) -> Iterable[Entity]:
        for ent in entity._.kb_ents:
            ent_str, _ = ent  # ent: (str, score)
            yield self.linker.kb.cui_to_entity[ent_str]

    def classify(
        self, answer: Dict[str, Any], document: Dict[str, Any]
    ) -> bool:

        doc_text = document["document.text"]
        answer_index = answer["answer.target"]
        answer_text = answer["answer.text"][answer_index]
        answer_synonyms = answer["answer.synonyms"]

        scispacy_doc = self.model(answer_text)

        answer_aliases = {answer["answer.text"][answer_index]}
        
        for entity in scispacy_doc.ents:
            e_aliases = set(self.extract_aliases(entity))
            answer_aliases = sorted(set.union(answer_aliases, answer_synonyms, e_aliases), key=len)

        # todo: investigate the aliases, some are too general
        rich.print(f">> aliases={answer_aliases}, count={len(answer_aliases)} doc_ents={scispacy_doc.ents}")

        # re.search: Scan through string looking for a location where the regular expression pattern produces a match, and return a corresponding MatchObject instance. Return None if no position in the string matches the pattern; note that this is different from finding a zero-length match at some point in the string.
        # re.escape: Return string with all non-alphanumerics backslashed; this is useful if you want to match an arbitrary literal string that may have regular expression metacharacters in it.
        # re.IGNORECASE: Perform case-insensitive matching
        return bool(
            re.search(
                re.compile('|'.join(re.escape(x) for x in answer_aliases), re.IGNORECASE)
                , doc_text)
        )

class ExactMatch(RelevanceClassifier):
    """Match the lower-cased answer string in the document."""

    def classify(
        self, answer: Dict[str, Any], document: Dict[str, Any]
    ) -> bool:
        doc_text = document["document.text"]
        answer_index = answer["answer.target"]
        answer_text = answer["answer.text"][answer_index]

        return bool(answer_text in doc_text)
