import os
from typing import Optional

import datasets
import faiss
import numpy as np
import rich
import torch
from torch import Tensor
from torch.utils import data
from torch.utils.data import DataLoader
from transformers import AutoModel
from transformers import AutoTokenizer

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# example
dataset = datasets.load_dataset("ptb_text_only", split="train")

# tokenize
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
dataset = dataset.map(
    lambda e: tokenizer(e["sentence"], truncation=True, padding="max_length"),
    batched=True,
)
dataset.set_format(
    type="torch",
    columns=["input_ids", "token_type_ids", "attention_mask", "sentence"],
)
small_dataset = dataset.select(range(1000))
rich.print(small_dataset["input_ids"].shape)

# bert model
model = AutoModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
model.eval()

with torch.no_grad():
    document_outputs = model(
        input_ids=small_dataset["input_ids"],
        attention_mask=small_dataset["attention_mask"],
    )
document_encoded_layers = document_outputs[0]

# faiss operations
d = document_encoded_layers.shape[2]  # dimension
nlist = 3  # number of clusters

quantiser = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantiser, d, nlist, faiss.METRIC_L2)

print(f"Index is trained: {index.is_trained}")
index.train(np.ascontiguousarray(document_encoded_layers[:, 0, :].numpy()))
print(f"Index is trained: {index.is_trained}")
index.add(np.ascontiguousarray(document_encoded_layers[:, 0, :].numpy()))
print(f"Total number of indices: {index.ntotal}")

k = 3  # number of nearest neighbours
query = tokenizer("he is afraid of getting lung cancer")
query_embeddings = model(
    input_ids=query["input_ids"], attention_mask=query["attention_mask"]
)
xq = query_embeddings


# Perform search on index
distances, indices = index.search(xq, k)
