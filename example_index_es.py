from fz_openqa.datamodules.medqa_dm import MedQaDataModule
import numpy as np
import rich, json, os

from fz_openqa.datamodules.corpus_dm import MedQaEnDataModule, FzCorpusDataModule
from fz_openqa.datamodules.medqa_dm import MedQaDataModule
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer

from fz_openqa.utils.run_elasticsearch import *

tokenizer = init_pretrained_tokenizer(pretrained_model_name_or_path='bert-base-cased')

corpus = MedQaEnDataModule(tokenizer=tokenizer,
                            passage_length=200,
                            passage_stride=100,
                            append_document_title=False,
                            num_proc=4,
                            use_subset=True,
                            verbose=False)
corpus.prepare_data()
corpus.setup()

print(f">> get corpus")
rich.print(corpus.dataset["train"])

print(f">> indexing the dataset using vectors")
corpus.build_index(model=lambda batch:  batch['document.input_ids'], index_mode="faiss")

print(f">> indexing the dataset using bm25")
corpus.build_index(index_mode="bm25")
print(f">> Indexing finished! See index here:")
rich.print("http://localhost:5601")

qst = "What is the symptoms of post polio syndrome?"
rich.print(qst)
hits = corpus.search_index(query=qst, index="bm25", k=1)

print(f">> Query response")
rich.print(hits)
