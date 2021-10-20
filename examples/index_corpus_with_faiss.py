import sys

import datasets
import torch

from examples.utils import display_search_results
from examples.utils import gen_example_query
from fz_openqa.datamodules.corpus_dm import FzCorpusDataModule
from fz_openqa.datamodules.index import FaissIndex
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer

datasets.set_caching_enabled(False)
print(f">> {sys.version}")

tokenizer = init_pretrained_tokenizer(
    pretrained_model_name_or_path="bert-base-cased"
)

query = gen_example_query(tokenizer)

dm = FzCorpusDataModule(
    tokenizer=tokenizer,
    index=FaissIndex(),
    verbose=True,
    num_proc=1,
    use_subset=True,
    train_batch_size=3,
)
dm.prepare_data()
dm.setup()

print(">> indexing the dataset using Faiss")


def model(batch):
    shape = (
        batch["document.input_ids"].size(0),
        10,
    )
    return torch.randn(shape)


dm.build_index(model=model)

print(">> query the dataset using Faiss")


def model(batch):
    shape = (
        batch["question.input_ids"].size(0),
        10,
    )
    return torch.randn(shape)


result = dm.search_index(query, k=3, model=model)

display_search_results(query, result)
