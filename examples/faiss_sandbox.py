import os

import datasets
import faiss
import numpy as np
import rich
import torch
from transformers import AutoModel
from transformers import AutoTokenizer

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
bert_id = "google/bert_uncased_L-2_H-128_A-2"

# load a text dataset
dataset = datasets.load_dataset("ptb_text_only", split="train")

# tokenize and format
tokenizer = AutoTokenizer.from_pretrained(bert_id)
dataset = dataset.map(
    lambda e: tokenizer(e["sentence"], truncation=True, padding="max_length"),
    batched=True,
)
dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask"],
)
small_dataset = dataset.select(range(1000))
rich.print(small_dataset)

# load the bert model model
model = AutoModel.from_pretrained(bert_id)
model.eval()

# computer vector representations of documents using BERT
with torch.no_grad():
    batch = small_dataset[:]
    batch = tokenizer.pad(batch)
    document_representations = model(**batch).last_hidden_state
    # keep only the CLS vector for each document
    document_representations = document_representations[:, 0, :]

    # convert to contiguous numpy array
    document_representations = document_representations.numpy()
    document_representations = np.ascontiguousarray(document_representations)

rich.print(f"Document representations: {document_representations.shape}")

# faiss parameters + init
ndims = document_representations.shape[-1]
nlist = 10  # number of clusters
quantiser = faiss.IndexFlatL2(ndims)
index = faiss.IndexIVFPQ(quantiser, ndims, nlist, 4, 4, faiss.METRIC_L2)

# attempting to add the vectors to the index
rich.print(f"Index is trained: {index.is_trained}")
index.train(document_representations)  # <- this line throws segmentation fault
rich.print(f"Index is trained: {index.is_trained}")
index.add(document_representations)
rich.print(f"Total number of indices: {index.ntotal}")


k = 3
query = tokenizer("he is afraid of getting lung cancer")
xq = model(
    input_ids=query["input_ids"], attention_mask=query["attention_mask"]
)
xq = xq.last_hidden_state[:, 0, :].numpy()

# Perform search on index
distances, indices = index.search(xq, k)
