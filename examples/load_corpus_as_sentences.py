from pprint import pprint

import datasets
from utils import display_search_results
from utils import gen_example_query

from fz_openqa.datamodules.corpus_dm import MedQaCorpusDataModule
from fz_openqa.datamodules.index import ElasticSearchIndex
from fz_openqa.datamodules.pipes import SearchCorpus
from fz_openqa.datamodules.pipes import TextFormatter
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer

es_body = {
    "settings": {
        # Shards are used to parallelize work on an index
        "number_of_shards": 1,
        # Replicas are copies of the shards and provide reliability if a node is lost
        "number_of_replicas": 0,
        #
        "similarity": {
            "default": {
                # By default, b has a value of 0.75 and k1 a value of 1.2
                "type": "BM25",
                # texts which touch on several topics often benefit by choosing a larger b
                # most experiments seem to show the optimal b to be in a range of 0.3-0.9
                "b": 0.75,
                # should generally trend toward larger numbers when the text is a long and diverse
                # most experiments seem to show the optimal k1 to be in a range of 0.5-2.0
                "k1": 1.2,
            }
        },
        # Defines changes to the text before tokenization and indexing
        "analysis": {
            "analyzer": {
                "custom_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    # token filters
                    "filter": [
                        # Converts tokens to lowercase
                        "lowercase",
                        # Removes tokens equivalent to english stopwords
                        "stop",
                        # Converts a-z, 1-9, and symbolic characters to their ASCII equivalent
                        "asciifolding",
                        # Converts tokens to its root word based snowball stemming
                        "my_snow",
                    ],
                }
            },
            "filter": {"my_snow": {"type": "snowball", "language": "English"}},
        },
    },
    # defining a mapping will help: (1) optimize the performance, (2) save disk space
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "title": {
                # Prevents the inverted index and doc values from being created
                "enabled": False
            },
            "idx": {"type": "integer"},
        }
    },
}

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
    es_body=es_body,
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
