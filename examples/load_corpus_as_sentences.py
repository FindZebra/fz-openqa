from pprint import pprint

import datasets
from utils import display_search_results
from utils import gen_example_query

from fz_openqa.datamodules.corpus_dm import MedQaCorpusDataModule
from fz_openqa.datamodules.index import ElasticSearchIndex
from fz_openqa.datamodules.pipes import SearchCorpus
from fz_openqa.datamodules.pipes import TextFormatter
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer

datasets.set_caching_enabled(False)

tokenizer = init_pretrained_tokenizer(
    pretrained_model_name_or_path="bert-base-cased"
)

query = gen_example_query(tokenizer)

es = ElasticSearchIndex(
    index_key="document.row_idx",
    text_key="document.text",
    query_key="question.text",
    num_proc=4,
    filter_mode=None,
    text_cleaner=TextFormatter(remove_symbols=True),
    analyze=False,
)

# load the corpus object
dm = MedQaCorpusDataModule(
    tokenizer=tokenizer,
    to_sentences=True,
    index=es,
    verbose=True,
    num_proc=4,
    use_subset=True,
    train_batch_size=3,
)

# prepare both the QA dataset and the corpus
dm.prepare_data()
dm.setup()

print(">> indexing the dataset using Elastic Search")
dm.build_index()

print(">> query the dataset using Elastic Search")
pipe = SearchCorpus(corpus_index=dm._index, k=5)
result = pipe(query)

display_search_results(corpus=dm.dataset, queries=query, results=result)
