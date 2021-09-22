from datetime import datetime

import rich
from elasticsearch import Elasticsearch
from elasticsearch import helpers
from numpy import array
from torch.functional import Tensor

"""
Deprecated:
OBS: remember to run the following line of code before execution:

'docker compose up'

with the supplied docker-compose.yml file to start two containers (ElasticSearch and Kibana, both version  7.13)

"""

es_config = {
    "settings": {
        "number_of_shards": 1,
        "analysis": {
            "analyzer": {
                "stop_standard": {
                    "type": "standard",
                    " stopwords": "_english_",
                }
            }
        },
    },
    "mappings": {
        "properties": {
            "ducment.attention_mask": {"type": "dense_vector", "dims": 2},
            "document.idx": {"type": "dense_vector", "dims": 1},
            "ducment.input_ids": {"type": "dense_vector", "dims": 2},
            "document.text": {
                "type": "text",
                "analyzer": "standard",
                "similarity": "BM25",
            },
            "docment.passage_idx": {"type": "dense_vector", "dims": 1},
            "document.passage_mask": {"type": "dense_vector", "dims": 2},
            "document.vectors": {"type": "dense_vector", "dims": 2},
        }
    },
}

es = Elasticsearch(timeout=60)  # ElasticSearch instance


def es_create_index(index_name: str):
    """
    Create ElasticSearch Index
    """
    es.indices.delete(index=index_name, ignore=[400, 404])
    response = es.indices.create(index=index_name)
    print(response)


def es_remove_index(index_name: str):
    """
    Remove ElasticSearch Index
    """
    response = es.indices.delete(index=index_name)
    print(response)


def es_ingest(index_name: str, title: str, paragraph: str):
    """
    Ingest to ElasticSearch Index
    """
    doc = {"title": title, "text": paragraph}
    response = es.create(
        index=index_name, body=doc, refresh="true", timeout=60
    )
    print(response)


def es_bulk(
    index_name: str, title: str, document_idx: list, document_txt: list
):
    actions = [
        {
            "_index": index_name,
            "_title": title,
            "_source": {
                "title": title,
                "idx": document_idx[i],
                "text": document_txt[i],
            },
        }
        for i in range(len(document_txt))
    ]

    response = helpers.parallel_bulk(
        es, actions, chunk_size=1000, request_timeout=200, refresh="true"
    )
    indices = []
    for succes, info in response:
        if succes:
            indices.append(info["index"]["_id"])

    return indices


def es_search(index_name: str, query: str, results: int):
    """
    Search in ElasticSearch Index
    """
    response = es.search(
        index=index_name,
        body={
            "query": {"match": {"text": query.lower()}},
            "from": 0,
            "size": results,
        },
    )

    return response[
        "hits"
    ]  # (object) Contains returned documents and metadata.
