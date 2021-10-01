import warnings
from typing import List

from elasticsearch import Elasticsearch
from elasticsearch import helpers
from elasticsearch.exceptions import RequestError


class ElasticSearch:
    def __init__(self, timeout=60):

        super().__init__()
        self.es = Elasticsearch(timeout=timeout)  # ElasticSearch instance

    def es_create_index(self, index_name: str) -> bool:
        """
        Create ElasticSearch Index
        """
        # todo @MotzWanted: don't override the dataset if existing.
        #  The index is generated given the dataset fingerprint, and should be unique.

        try:
            # self.es.indices.delete(index=index_name, ignore=[400, 404])
            _ = self.es.indices.create(index=index_name)
            created = True

        # todo: handle specific exceptions
        except RequestError as err:
            warnings.warn(f"{err}")
            created = False

        return created

    def es_remove_index(self, index_name: str):
        """
        Remove ElasticSearch Index
        """
        return self.es.indices.delete(index=index_name)

    def es_ingest(self, index_name: str, title: str, paragraph: str):
        """
        Ingest to ElasticSearch Index
        """
        doc = {"title": title, "text": paragraph}
        return self.es.create(
            index=index_name, body=doc, refresh="true", timeout=60
        )

    def es_bulk(
        self,
        index_name: str,
        title: str,
        document_idx: list,
        document_txt: list,
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

        _ = helpers.bulk(
            self.es,
            actions,
            chunk_size=1000,
            request_timeout=200,
            refresh="true",
        )

    def es_search_bulk(self, index_name: str, queries: List[str], k: int):
        """
        Batch search in ElasticSearch Index
        """
        request = []
        req_head = [{"index": index_name}] * len(queries)
        req_body = [
            {
                "query": {"match": {"text": queries[i]}},
                "from": 0,
                "size": k,
            }
            for i in range(len(queries))
        ]

        request = [
            item for sublist in zip(req_head, req_body) for item in sublist
        ]

        result = self.es.msearch(body=request)

        indexes, scores = [], []
        for query in result["responses"]:
            temp_indexes, temp_scores = [], []
            for hit in query["hits"]["hits"]:
                temp_scores.append(hit["_score"])
                temp_indexes.append(hit["_source"]["idx"])
            indexes.append(temp_indexes)
            scores.append(temp_scores)

        return scores, indexes

    def es_search(self, index_name: str, query: str, results: int):
        """
        Sequential search in ElasticSearch Index
        """
        response = self.es.search(
            index=index_name,
            body={
                "query": {"match": {"text": query}},
                "from": 0,
                "size": results,
            },
        )

        return response[
            "hits"
        ]  # (object) Contains returned documents and metadata.
