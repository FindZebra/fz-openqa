import os
import sys

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import datasets
from utils import display_search_results, gen_example_query

from fz_openqa.datamodules.corpus_dm import FzCorpusDataModule
from fz_openqa.datamodules.index import ElasticSearchIndex
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer

datasets.set_caching_enabled(False)
print(f">> {sys.version}")

tokenizer = init_pretrained_tokenizer(
    pretrained_model_name_or_path="bert-base-cased"
)

query = gen_example_query(tokenizer)

dm = FzCorpusDataModule(
    tokenizer=tokenizer,
    index=ElasticSearchIndex(
        index_key="document.row_idx",
        text_key="document.text",
        query_key="question.text",
        num_proc=4,
        filter_mode="stopwords",
    ),
    verbose=True,
    num_proc=4,
    use_subset=True,
    train_batch_size=3,
)
dm.prepare_data()
dm.setup()

print(">> indexing the dataset using Elastic Search")
dm.build_index()

print(">> query the dataset using Elastic Search")
result = dm.search_index(query, k=3)
print(query)
print(dm.dataset[8])
display_search_results(query, result)
