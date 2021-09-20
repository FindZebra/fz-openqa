# poetry run pip install scispacy
# poetry run pip install spacy
# poetry run pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_sm-0.4.0.tar.gz
# poetry run pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_md-0.4.0.tar.gz
from pprint import pprint

import en_core_sci_md
import en_core_sci_sm
import scispacy
import spacy
from spacy import displacy


def display_entities(model, document):
    """A function that returns a tuple of displacy omage of named or unnamed word entities and a set of unique entities recognized based on scispacy model in use
    Args:
        model: A pretrained model from spaCy or scispaCy
        document: text data to be analysed
    """
    nlp = model.load()
    doc = nlp(document)
    displacy_image = displacy.render(
        doc, jupyter=True, style="ent"
    )  # it is only possible to render HTML in a browser or jupyter notebook
    entity_and_label = pprint(set([(X.text, X.label_) for X in doc.ents]))

    return (displacy_image, entity_and_label)


doc2 = "the treatment of Lyme disease.After appropriately treated Lyme disease, a small percentage of patients continue to have subjective symptoms, primarily musculoskeletal pain, neurocognitive difficulties, or fatigue. This chronic Lyme disease or post–Lyme syndrome is sometimes a disabling condition that is similar to chronic fatigue syndrome or fibromyalgia. In a large study, one group of patients with post–Lyme syndrome received IV ceftriaxone for 30 days followed by oral doxycycline for 60 days, while another group received IV and oral placebo preparations for the same durations. No significant differences were found between groups in the numbers of patients reporting that their symptoms had improved, become worse, or stayed the same. Such patients are best treated for the relief of symptoms rather than with prolonged courses of antibiotics.The risk of infection with B. burgdorferi after a recognized tick bite is so low that antibiotic"

display_entities(en_core_sci_md, doc2)
