import math
import string
from typing import Dict
from typing import List
from typing import Optional

import rich
import torch
from elasticsearch import Elasticsearch
from elasticsearch import helpers as es_helpers
from elasticsearch import RequestError
from loguru import logger


def tokenize(text: str) -> List[str]:
    """
    Filter Punctuation
    """
    tokens = text.split(" ")
    tokens = [token.translate(str.maketrans("", "", string.punctuation)) for token in tokens]
    tokens = [token for token in tokens if token.strip() != ""]
    return tokens


def es_search_bulk(
    es_instance: Elasticsearch,
    *,
    index_name: str,
    queries: List[str],
    auxiliary_queries: List[str] = None,
    document_ids: List[int] = None,
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
        use_aux_queries = auxiliary_queries is not None and auxiliary_weight > 0
        should_query_parts = []
        filter_query = []

        # measure the query and the auxiliary query
        query_length = len(tokenize(query))
        if auxiliary_queries is not None:
            aux_query = auxiliary_queries[i]
            aux_query_length = len(tokenize(aux_query))
        else:
            aux_query = None
            aux_query_length = None

        # this is the main query
        should_query_parts.append(
            {
                "match": {
                    "text": {
                        "query": query,
                        # "zero_terms_query": "all",
                        "operator": "or",
                    }
                }
            },
        )

        # this is an additional query term using the auxiliary_queries (answer option)
        if use_aux_queries:
            if auxiliary_weight > 0 and aux_query_length > 0:
                r = query_length / aux_query_length
                aux_weight_i = max(r, 1)
                aux_weight_i = auxiliary_weight * math.log(aux_weight_i)
                aux_weight_i = max(aux_weight_i, 0)
            else:
                aux_weight_i = 0
            should_query_parts.append(
                {
                    "match": {
                        "text": {
                            "query": aux_query,
                            "operator": "or",
                            "boost": aux_weight_i,
                        }
                    }
                },
            )

        if document_ids is not None:
            doc_id = document_ids[i]
            if isinstance(doc_id, torch.Tensor):
                doc_id = doc_id.item()
            filter_query.append({"term": {"document_idx": int(doc_id)}})

        # final request
        r = {
            "query": {
                "bool": {"should": should_query_parts, "filter": filter_query},
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
        if "hits" not in query:
            rich.print("[magenta]===== ES RESPONSE =====")
            rich.print(query)
            rich.print("[magenta]=======================")
            raise ValueError("ES did not return any hits (see above for details)")

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
    row_idx: List[int],
    document_txt: List[str],
    document_idx: List[int] = None,
    title: str = "__no_title__",
    chunk_size=1000,
    request_timeout=200,
):
    actions = [
        {
            "_index": index_name,
            "_title": title,
            "_source": {
                "title": title,
                "idx": row_idx[i],
                "text": document_txt[i],
                "document_idx": int(document_idx[i]) if document_idx is not None else None,
            },
        }
        for i in range(len(document_txt))
    ]

    return es_helpers.bulk(
        es_instance,
        actions,
        chunk_size=chunk_size,
        request_timeout=request_timeout,
        refresh="true",
    )
