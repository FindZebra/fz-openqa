from fz_openqa.datamodules.medqa_dm import MedQaDataModule
import numpy as np
import rich, json, os

from fz_openqa.datamodules.corpus_dm import MedQaEnDataModule, FzCorpusDataModule
from fz_openqa.datamodules.medqa_dm import MedQaDataModule
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer
from torch.utils.data import DataLoader

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
#print(corpus.dataset['train'])

print(f">> get corpus")
rich.print(corpus.dataset["train"])

print(f">> indexing the dataset using vectors")
corpus.index(model=lambda batch:  batch['document.input_ids'], index="faiss")

print(f">> indexing the dataset using bm25")
corpus.index(index="bm25")
print(f">> Indexing finished! See index here:")
rich.print("http://localhost:5601")

qst = "What is the symptoms of post polio syndrome?"
rich.print(qst)
hits = corpus.search_index(query=qst, index="bm25", k=1)

print(f">> Query response")
rich.print(hits)

questions_dm = MedQaDataModule(append_document_title=False,
                            tokenizer=tokenizer,
                            num_proc=4,
                            use_subset=False,
                            top_n_synonyms=3,
                            verbose=False)

questions_dm.prepare_data()
questions_dm.setup()
print(questions_dm.dataset)


# 1. sample data
#batch = next(iter(questions_dm.train_dataloader())) #


"""
batch = {
+ question.text: list of N texts,
+ question.input_ids:tensor of shape [N, L_q],
+ answer.text: N lists of 4 texts,
+ answer.input_ids: tensor of shape [N, 4, L_a],
+ answer.target: tensor of shape [N,],
}
"""

# 2. compute the dense representation (optional for bm25)
#print(f">> indexing the dataset using vectors")
#corpus.index(model=lambda batch:  batch['document.input_ids'], index="faiss")

# 3. index the corpus
#print(f">> indexing the dataset using bm25")
#corpus.index(index="bm25")
#print(f">> Indexing finished! See index here:")
#rich.print("http://localhost:5601")

# 4. query the data
#returned_indexes = corpus.search_index(index="bm25", k=100)
# implement in search_index corpus
#batch = corpus.collate_fn(
#    (corpus.dataset["train"][idx] for idx in returned_indexes))


# 5. classify the docuements

#print(f">> Get questions")
#rich.print(questions.dataset['train'])

#print(f">> querying MedQA questions")
#out, discarded = corpus.exact_method(queries=questions.dataset['train']['question.text'],
#                            answers=questions.dataset['train']['answer.text'],
#                            answer_idxs=questions.dataset['train']['answer.target'],
#                            synonyms=questions.dataset['train']['synonyms']
#                            )

#print(f">> Excact match output")
#rich.print(out['data'][0])
#print(f">> Number of mapped questions")
#rich.print(len(out['data']))

#print(f">> Number of discarded questions")
#rich.print(len(discarded['data']))
