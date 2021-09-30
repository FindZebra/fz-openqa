#from fz_openqa.datamodules.medqa_dm import MedQaDataModule
import numpy as np
import rich, json, os

from fz_openqa.datamodules.corpus_dm import MedQaCorpusDataModule, FzCorpusDataModule
#from fz_openqa.datamodules.medqa_dm import MedQaDataModule
from fz_openqa.utils.es_functions import ElasticSearch
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer

#from fz_openqa.utils.run_elasticsearch import *

tokenizer = init_pretrained_tokenizer(pretrained_model_name_or_path='bert-base-cased')

corpus = FzCorpusDataModule(tokenizer=tokenizer,
                               passage_length=200,
                               passage_stride=100,
                               append_document_title=False,
                               num_proc=4,
                               use_subset=True,
                               verbose=False)
corpus.prepare_data()
corpus.setup()

data = corpus.dataset

es = ElasticSearch()
es.es_create_index("corpus")
_ = es.es_bulk(
    index_name = "corpus", 
    title="book1", 
    document_idx=data['document.idx'],
    passage_idx=data['document.passage_idx'], 
    document_txt=data['document.text'])

qst = [
    "What is the symptoms of post polio syndrome?",
    "How is cancer cured?",
    "What is Parkinson syndrome?",
    "What is the symptoms of influenza?",
    "What is the president of united states?"
]

indexes = es.es_search_bulk(index_name="corpus", queries=qst, k=3)

print(f">> Query response")
rich.print(indexes)
