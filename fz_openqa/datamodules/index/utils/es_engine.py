import logging
from typing import Dict
from typing import List
from typing import Optional

from elasticsearch import Elasticsearch
from elasticsearch import helpers
from elasticsearch.client.indices import IndicesClient
from elasticsearch.exceptions import RequestError

logger = logging.getLogger(__name__)


def ping_es() -> bool:
    return Elasticsearch().ping()


class ElasticSearchEngine:
    _instance: Elasticsearch
    _indices_client: Optional[IndicesClient] = None

    def __init__(self, timeout=60, analyze=False):

        super().__init__()
        self.timeout = timeout
        self.analyze = analyze
        self._instance = Elasticsearch(timeout=self.timeout)
        if self.analyze:
            self._indices_client = IndicesClient(self._instance)

    def __getstate__(self):
        """this method is called when attempting pickling.
        ES instances cannot be properly pickled"""
        state = self.__dict__.copy()
        # Don't pickle the ES instances
        for attr in ["_instance", "_indices_client"]:
            if attr in state:
                state.pop(attr)

        return state

    def __setstate__(self, state):
        state["_instance"] = Elasticsearch(timeout=state["timeout"])
        if state["analyze"]:
            state["_indices_client"] = IndicesClient(state["_instance"])

        self.__dict__ = state

    @property
    def instance(self):
        return self._instance

    def es_create_index(self, index_name: str, body: Optional[Dict] = None) -> bool:
        """
        Create ElasticSearch Index
        """
        try:
            response = self.instance.indices.create(index=index_name, body=body)
            logger.info(response)
            newly_created = True

        except RequestError as err:
            if err.error == "resource_already_exists_exception":
                logger.info(f"ElasticSearch index with name=`{index_name}` already exists.")
                newly_created = False
            else:
                raise err

        return newly_created

    def es_remove_index(self, index_name: str):
        """
        Remove ElasticSearch Index
        """
        return self.instance.indices.delete(index=index_name)

    def es_ingest(self, index_name: str, title: str, paragraph: str):
        """
        Ingest to ElasticSearch Index
        """
        doc = {"title": title, "text": paragraph}
        return self.instance.create(index=index_name, body=doc, refresh="true", timeout=60)

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
            self.instance,
            actions,
            chunk_size=1000,
            request_timeout=200,
            refresh="true",
        )

    def es_search_bulk(self, index_name: str, queries: List[str], k: int):
        """
        Batch search in ElasticSearch Index
        """

        req_head = [{"index": index_name}] * len(queries)
        req_body = [
            {
                "query": {"match": {"text": {"query": queries[i], "zero_terms_query": "all"}}},
                "from": 0,
                "size": k,
            }
            for i in range(len(queries))
        ]

        request = [item for sublist in zip(req_head, req_body) for item in sublist]

        result = self.instance.msearch(body=request)

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

    def es_search(self, index_name: str, query: str, results: int):
        """
        Sequential search in ElasticSearch Index
        """
        response = self.instance.search(
            index=index_name,
            body={
                "query": {"match": {"text": query}},
                "from": 0,
                "size": results,
            },
        )

        return response["hits"]  # (object) Contains returned documents and metadata.

    def es_analyze_text(self, index_name: str, queries: List[str]):
        analyzed_tokens = []
        for docs in queries:
            results = [
                self._indices_client.analyze(
                    index=index_name,
                    body={"analyzer": "custom_analyzer", "text": doc},
                )
                for doc in docs
            ]
            temp_analysed = []
            for res in results:
                temp_analysed.append([term["token"] for term in res["tokens"]])
            analyzed_tokens.append(temp_analysed)

        return analyzed_tokens
