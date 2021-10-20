import os
import sys

import rich

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from fz_openqa.datamodules.corpus_dm import FzCorpusDataModule  # noqa: E402
from fz_openqa.tokenizers.pretrained import (
    init_pretrained_tokenizer,
)  # noqa: E402
from fz_openqa.utils.es_functions import ElasticSearchEngine  # noqa: E402

tokenizer = init_pretrained_tokenizer(
    pretrained_model_name_or_path="bert-base-cased"
)

corpus = FzCorpusDataModule(
    tokenizer=tokenizer,
    passage_length=200,
    passage_stride=100,
    append_document_title=False,
    num_proc=4,
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
    document_idx=data["idx"],
    document_txt=data["document.text"],
)

qst = [
    "What is the symptoms of post polio syndrome?",
    "How is cancer cured?",
    "What is Parkinson syndrome?",
    "What is the symptoms of influenza?",
    "What is the president of united states?",
]

output = es.es_search_bulk(index_name="corpus", queries=qst, k=3)

print(">> Query response")
rich.print(output)

rich.print(data[344])
rich.print(data[190])
rich.print(data[2])
