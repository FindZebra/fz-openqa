import datasets
import faiss.contrib.torch_utils
import numpy as np
import rich
import torch
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

# load the corpus object
corpus = MedQaCorpusDataModule(
    tokenizer=tokenizer,
    index=ElasticSearchIndex(
        index_key="document.row_idx",
        text_key="document.text",
        query_key="question.metamap",
        num_proc=4,
        filter_mode=None,
    ),
    verbose=False,
    num_proc=4,
    use_subset=True,
    passage_length=200,
    max_length=None,
)
corpus.prepare_data()
corpus.setup()
batch_corpus = next(iter(corpus.train_dataloader()))
pprint_batch(batch_corpus)

# load the QA dataset
dm = MedQaDataModule(
    tokenizer=tokenizer,
    train_batch_size=100,
    num_proc=4,
    num_workers=4,
    use_subset=True,
    verbose=True,
    corpus=corpus,
    relevance_classifier=ExactMatch(),
    compile_in_setup=False,
)

dm.subset_size = [500, 100, 100]
dm.prepare_data()
dm.setup()
batch_dm = next(iter(dm.train_dataloader()))
pprint_batch(batch_dm)


with torch.no_grad():
    document_outputs = model(
        input_ids=batch_corpus["document.input_ids"],
        attention_mask=batch_corpus["document.attention_mask"],
    )
    document_encoded_layers = document_outputs[0]

    question_outputs = model(
        input_ids=dm.dataset["train"]["question.input_ids"],
        attention_mask=dm.dataset["train"]["question.attention_mask"],
    )
    question_encoded_layers = question_outputs[0]

rich.print("Document encoded layers:", document_encoded_layers[:, 0, :])
rich.print("Question encoded layers:", question_encoded_layers[:, 0, :])


# d = batch["document.input_ids"].shape[1]
# nlist = 2  # number of clusters
# index = faiss.IndexFlatL2(d)
# quantiser = faiss.IndexFlatL2(d)
# index = faiss.IndexIVFFlat(quantiser, d, nlist, faiss.METRIC_L2)

# print(batch["document.input_ids"])

# print(f"Index is trained: {index.is_trained}")
# index.train(batch["document.input_ids"].type(torch.float32))
# print(f"Index is trained: {index.is_trained}")
# index.add(batch["document.input_ids"].type(torch.float32))
# print(f"Total number of indices: {index.ntotal}")

# rich.print(type(dm.dataset["train"]["question.input_ids"]))

# k = 4
# xq_numpy = np.array(dm.dataset["train"]["question.input_ids"])
# xq = np.expand_dims(xq_numpy, axis=0)
# rich.print(xq.shape, batch["document.input_ids"].shape)
# distances, indices = index.search(xq, k)
# print(indices)
