from copy import deepcopy

import datasets
import faiss.contrib.torch_utils
import numpy as np
import rich
import torch
from torch.functional import Tensor
from transformers import AutoModel

from fz_openqa.datamodules.corpus_dm import MedQaCorpusDataModule
from fz_openqa.datamodules.index import ElasticSearchIndex
from fz_openqa.datamodules.meqa_dm import MedQaDataModule
from fz_openqa.datamodules.pipes import ExactMatch
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.pretty import get_separator
from fz_openqa.utils.pretty import pprint_batch
from fz_openqa.utils.train_utils import setup_safe_env


datasets.set_caching_enabled(True)
setup_safe_env()

tokenizer = init_pretrained_tokenizer(
    pretrained_model_name_or_path="bert-base-cased"
)

model = AutoModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
model.eval()


def encode_bert(
    document_input_ids: Tensor,
    document_attention_mask: Tensor,
    question_input_ids: Tensor,
    question_attention_mask: Tensor,
):
    """
    Compute document and question embeddings using BERT
    """
    with torch.no_grad():
        document_outputs = model(
            input_ids=document_input_ids,
            attention_mask=document_attention_mask,
        )
        document_encoded_layers = document_outputs[0]

        question_outputs = model(
            input_ids=question_input_ids,
            attention_mask=question_attention_mask,
        )
        question_encoded_layers = question_outputs[0]

    return document_encoded_layers, question_encoded_layers


# load the corpus object
corpus = MedQaCorpusDataModule(
    tokenizer=tokenizer,
    index=ElasticSearchIndex(
        index_key="document.row_idx",
        text_key="document.text",
        query_key="question.metamap",
        num_proc=1,
        filter_mode=None,
    ),
    verbose=False,
    num_proc=1,
    use_subset=True,
    passage_length=200,
    max_length=None,
    train_batch_size=200,
)
corpus.prepare_data()
corpus.setup()
batch_corpus = next(iter(corpus.train_dataloader()))
pprint_batch(batch_corpus)

# load the QA dataset
dm = MedQaDataModule(
    tokenizer=tokenizer,
    train_batch_size=200,
    num_proc=1,
    num_workers=1,
    use_subset=True,
    verbose=True,
    corpus=corpus,
    relevance_classifier=ExactMatch(),
    compile_in_setup=False,
)

dm.prepare_data()
dm.setup()
batch_dm = next(iter(dm.train_dataloader()))
pprint_batch(batch_dm)


# Compute document and question embeddings
document_encoded_layers, question_encoded_layers = encode_bert(
    document_input_ids=torch.tensor(batch_corpus["document.input_ids"]),
    document_attention_mask=torch.tensor(
        batch_corpus["document.attention_mask"]
    ),
    question_input_ids=batch_dm["question.input_ids"].clone().detach(),
    question_attention_mask=batch_dm["question.attention_mask"]
    .clone()
    .detach(),
)


d = document_encoded_layers.shape[2]  # dimension
rich.print(d)
nlist = 5  # number of clusters
index = faiss.IndexFlatL2(d)
# quantiser = faiss.IndexFlatL2(d)
# index = faiss.IndexIVFFlat(quantiser, d, nlist, faiss.METRIC_L2)


# print(f"Index is trained: {index.is_trained}")
# index.train(document_encoded_layers[:, 0, :].contiguous())
print(f"Index is trained: {index.is_trained}")
index.add(document_encoded_layers[:10, 0, :].contiguous())
print(f"Total number of indices: {index.ntotal}")

# rich.print(question_encoded_layers[1:5, 0, :].contiguous().shape)
k = 3  # number of nearest neighbours
xq = deepcopy(
    torch.zeros_like(question_encoded_layers[1:2, 0, :].contiguous())
)

# Perform search on index
distances, indices = index.search(xq, k)
print(indices)
