import os
import sys

import rich

from fz_openqa.utils.pretty import get_separator

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from fz_openqa.datamodules.corpus_dm import FzCorpusDataModule
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer
from fz_openqa.datamodules.index.utils.es_engine import ElasticSearchEngine

tokenizer = init_pretrained_tokenizer(
    pretrained_model_name_or_path="bert-base-cased"
)

corpus = FzCorpusDataModule(
    tokenizer=tokenizer,
    use_subset=True,
    verbose=False,
)
corpus.prepare_data()
corpus.setup()

data = corpus.dataset

es = ElasticSearchEngine()
es.es_create_index("corpus")
_ = es.es_bulk(
    index_name="corpus",
    title="book1",
    document_idx=data["document.row_idx"],
    document_txt=data["document.text"],
)

qst = [
    "What is the symptoms of post polio syndrome?",
    "How is cancer cured?",
    "What is Parkinson syndrome?",
    "What is the symptoms of influenza?",
    "What is the president of united states?",
]

score, index = es.es_search_bulk(index_name="corpus", queries=qst, k=3)

print(get_separator())
rich.print(score)
print(get_separator())
rich.print(index)
