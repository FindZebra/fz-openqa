import warnings
from typing import Dict
from typing import List
from typing import Optional

from elasticsearch import Elasticsearch
from elasticsearch import helpers as es_helpers
from elasticsearch import RequestError
from loguru import logger


def es_search_bulk(
    es_instance: Elasticsearch,
    *,
    index_name: str,
    queries: List[str],
    auxiliary_queries: List[str] = None,
    auxiliary_weight: float = 0,
    k: int = 10,
):
    """
    Batch search in ElasticSearch Index
    """

    if auxiliary_queries is None and auxiliary_weight > 0:
        raise ValueError("auxiliary_queries must be provided " "if auxiliary_weight > 0")

    request = []
    for i, query in enumerate(queries):

        # this is the main query
        query_parts = [
            {
                "match": {
                    "text": {
                        "query": query,
                        # "zero_terms_query": "all",
                        "operator": "or",
                    }
                }
            },
        ]

        # this is an additional query term using the auxiliary_queries (answer option)
        if auxiliary_queries is not None and auxiliary_weight > 0:
            query_parts.append(
                {
                    "match": {
                        "text": {
                            "query": auxiliary_queries[i],
                            "operator": "or",
                            "boost": auxiliary_weight,
                        }
                    }
                },
            )

        # final request
        r = {
            "query": {
                "bool": {"should": query_parts},
            },
            "from": 0,
            "size": k,
        }

        # append the header and body of the request
        request.extend([{"index": index_name}, r])

    result = es_instance.msearch(body=request, index=index_name, request_timeout=200)

    indexes, scores, contents = [], [], []
    for query in result["responses"]:
        temp_indexes, temp_scores, temp_content = [], [], []
        for hit in query["hits"]["hits"]:
            temp_scores.append(hit["_score"])
            temp_indexes.append(hit["_source"]["idx"])
            temp_content.append(hit["_source"]["text"])
        indexes.append(temp_indexes)
        scores.append(temp_scores)
        contents.append(temp_content)

    return scores, indexes, contents


def es_search(es_instance: Elasticsearch, *, index_name: str, query: str, k: int):
    """
    Sequential search in ElasticSearch Index
    """
    response = es_instance.search(
        index=index_name,
        body={
            "query": {"match": {"text": query}},
            "from": 0,
            "size": k,
        },
    )

    return response["hits"]  # (object) Contains returned documents and metadata.


def es_create_index(
    es_instance: Elasticsearch, index_name: str, body: Optional[Dict] = None
) -> bool:
    """
    Create ElasticSearch Index
    """
    try:
        response = es_instance.indices.create(index=index_name, body=body)
        logger.info(response)
        newly_created = True

    except RequestError as err:
        if err.error == "resource_already_exists_exception":
            logger.info(f"ElasticSearch index with name=`{index_name}` already exists.")
            newly_created = False
        else:
            raise err

    return newly_created


def es_remove_index(es_instance: Elasticsearch, index_name: str):
    """
    Remove ElasticSearch Index
    """
    return es_instance.indices.delete(index=index_name)


def es_ingest(es_instance: Elasticsearch, index_name: str, title: str, paragraph: str):
    """
    Ingest to ElasticSearch Index
    """
    doc = {"title": title, "text": paragraph}
    return es_instance.create(index=index_name, body=doc, refresh="true", timeout=60)


def es_ingest_bulk(
    es_instance: Elasticsearch,
    index_name: str,
    *,
    document_idx: List[int],
    document_txt: List[str],
    title: str = "__no_title__",
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

    return es_helpers.bulk(
        es_instance,
        actions,
        chunk_size=1000,
        request_timeout=200,
        refresh="true",
    )
