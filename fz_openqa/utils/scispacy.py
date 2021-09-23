# !poetry run pip install scispacy
# !poetry run pip install spacy
# !poetry run pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_sm-0.4.0.tar.gz
# !poetry run pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_md-0.4.0.tar.gz
# !poetry run pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_ner_bc5cdr_md-0.4.0.tar.gz
from pprint import pprint

import en_core_sci_md
import en_core_sci_sm
import en_ner_bc5cdr_md
import scispacy
import spacy
from spacy import displacy

from . import Pipe


def display_entities_pipe(model, document) -> Pipe:
    """
    Build a Pipe to return a tuple of displacy image of named or unnamed word entities and a set of unique entities recognized based on scispacy model in use
    Args:
        model: A pretrained model from spaCy or scispaCy
        document: text data to be analysed
    """
    nlp = model.load()
    doc = nlp(document)
    displacy_image = displacy.render(
        doc, jupyter=True, style="ent"
    )  # it is only possible to render HTML in a browser or jupyter notebook -> will return None
    entity_and_label = pprint(set([(X.text, X.label_) for X in doc.ents]))

    return (displacy_image, entity_and_label)
