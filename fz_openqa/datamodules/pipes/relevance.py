import re
from typing import Any
from typing import Dict
from typing import Iterable
from typing import Optional

import numpy as np
import spacy
import torch
from scispacy.linking_utils import Entity

from .static import DISCARD_TUIs
from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.utils.datastruct import Batch

np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


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
        batch_size = self.batch_size(batch)
        for i in range(batch_size):
            eg_ans_i = self.get_eg(
                batch,
                i,
                filter_op=lambda key: str(key).startswith(self.answer_prefix),
            )
            eg_doc_i = self.get_eg(
                batch,
                i,
                filter_op=lambda key: str(key).startswith(
                    self.document_prefix
                ),
            )

            # iterate through each document
            results_i = []
            n_docs = len(next(iter(eg_doc_i.values())))
            for j in range(n_docs):
                d_data_ij = {k: v[j] for k, v in eg_doc_i.items()}
                results_i += [self.classify(eg_ans_i, d_data_ij)]
            results += [results_i]
        results = torch.tensor(results)
        batch[self.output_key] = results

        return batch


class MetaMapMatch(RelevanceClassifier):
    def __init__(self, model_name: Optional[str] = "en_core_sci_lg", **kwargs):
        super().__init__()
        from scispacy.abbreviation import AbbreviationDetector  # type: ignore
        from scispacy.linking import EntityLinker  # type: ignore

        self.model_name = model_name
        self.model = spacy.load(self.model_name)
        self.model.add_pipe("abbreviation_detector")
        self.model.add_pipe(
            "scispacy_linker",
            config={
                "resolve_abbreviations": True,
                "linker_name": "umls",
                "max_entities_per_mention": 1,
            },
        )
        self.linker = self.model.get_pipe("scispacy_linker")

    def classify(
        self, answer: Dict[str, Any], document: Dict[str, Any]
    ) -> bool:
        doc_text = document["document.text"]
        answer_index = answer["answer.target"]
        answer_aliases = [answer["answer.text"][answer_index]]
        answer_cui = answer["answer.cui"][0]
        answer_aliases.extend(set(self.linker.kb.cui_to_entity[answer_cui][2]))
        return bool(
            re.findall(
                r"(?=(" + "|".join(answer_aliases) + r"))",
                doc_text,
                re.IGNORECASE,
            )
        )


class SciSpacyMatch(RelevanceClassifier):
    def __init__(self, model_name: Optional[str] = "en_core_sci_lg", **kwargs):
        super().__init__()
        from scispacy.abbreviation import AbbreviationDetector  # type: ignore
        from scispacy.linking import EntityLinker  # type: ignore

        self.model_name = model_name
        self.model = spacy.load(self.model_name)
        self.model.add_pipe("abbreviation_detector")
        self.model.add_pipe(
            "scispacy_linker",
            config={
                "resolve_abbreviations": True,
                "linker_name": "umls",
                "max_entities_per_mention": 1,
            },
        )
        self.linker = self.model.get_pipe("scispacy_linker")

    def extract_aliases(self, entity) -> Iterable[str]:
        # get the list of linked entity
        linked_entities = self.get_linked_entities(entity)

        # filter irrelevant entities based on TUIs
        def keep_entity(ent: Entity) -> bool:
            """
            keep the entity if at least one of the TUIs is not in
            the DISCARD_TUIs list.
            # todo: is that the right behaviour?
            """
            return any(tui not in DISCARD_TUIs for tui in ent.types)

        linked_entities = filter(lambda ent: keep_entity, linked_entities)

        # return aliases
        for linked_entity in linked_entities:
            for alias in linked_entity.aliases:
                yield alias

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

        scispacy_doc = self.model(answer_text)

        answer_aliases = {answer["answer.text"][answer_index]}
        for entity in scispacy_doc.ents:
            e_aliases = set(self.extract_aliases(entity))
            answer_aliases = set.union(answer_aliases, e_aliases)

        # todo: investigate the aliases, some are too general
        # rich.print(f">> aliases={answer_aliases}")

        return any(
            alias.lower() in doc_text.lower() for alias in answer_aliases
        )
        # todo: fix this: crashes with re.error: nothing to repeat at position 33
        # return bool(
        #     re.findall(
        #         r"(?=(" + "|".join(answer_aliases) + r"))",
        #         doc_text,
        #         re.IGNORECASE,
        #     )
        # )


class ExactMatch(RelevanceClassifier):
    """Match the lower-cased answer string in the document."""

    def classify(
        self, answer: Dict[str, Any], document: Dict[str, Any]
    ) -> bool:
        doc_text = document["document.text"]
        answer_index = answer["answer.target"]
        answer_text = answer["answer.text"][answer_index]

        return bool(answer_text.lower() in doc_text.lower())
