import itertools
from typing import List

from scispacy.linking import EntityLinker  # type: ignore
from spacy.language import Language
from spacy.tokens import Doc
from spacy.tokens.span import Span
from spacy.util import filter_spans


def _merge_entities(doc: Doc) -> List[Span]:
    """Merge doc text into a single entity.

    Parameters
    ----------
    doc
        (spacy.tokens.Doc): Doc object.

    Returns
    -------
    List[Span]
        span of merged entities
    """
    return [Span(doc, 0, doc.__len__(), label="Entity")]


def _consecutive_entities(doc: Doc) -> List[Span]:
    """Create list of spans of consecutive entities based on doc text.

    Parameters
    ----------
    doc
        (spacy.tokens.Doc): Doc object.

    Returns
    -------
    List[Span]
        spans of consecutive entities
    """
    # index recognized entities; split when entity holds multiple tokens
    entities = list(itertools.chain(*[ent.text.split() for ent in doc.ents]))
    spans = []
    # join the entities
    for ent in doc.ents:
        # identify consecutive entities in doc text
        for j in range(ent.start, doc.__len__()):
            # find index where consecutive entities ends
            if doc[j].text not in entities:
                spans.append(Span(doc, ent.start, doc[j].i, label="Entity"))
                break
        # if last word in doc text is an entity
        if doc[j].text in entities:
            spans.append(Span(doc, ent.start, doc.ents[-1].end, label="Entity"))
            break
    # filter sequence of spans and remove duplicates or overlaps.
    return filter_spans(spans)


@Language.component("merge_consecutive_entities")
def merge_consecutive_entities(doc: Doc) -> Doc:
    """Merge consecutive entities into a single entity if doc text holds more than 3 words;
    else merge the whole doc text into a single entity.

    Parameters
    ----------
    doc
        (spacy.tokens.Doc): Doc object.

    Returns
    ----------
    doc
        (spacy.tokens.Doc): Doc object with merged consecutive entities.
    """
    if doc.__len__() > 3:
        doc.ents = _consecutive_entities(doc=doc)
    else:
        doc.ents = _merge_entities(doc=doc)
    return doc
