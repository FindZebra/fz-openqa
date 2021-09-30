from pprint import pprint

from spacy import displacy


def display_entities_pipe(model, document):
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
