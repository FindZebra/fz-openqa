import warnings

from elasticsearch import Elasticsearch
from elasticsearch import helpers
from elasticsearch.exceptions import RequestError

class ElasticSearch():

    def __init__(
        self,
        es = Elasticsearch(timeout=60)  # ElasticSearch instance
    ):
    
        super().__init__()

        self.es = es

    def es_create_index(self, index_name: str) -> bool:
        """
        Create ElasticSearch Index
        """
        # todo @MotzWanted: don't override the dataset if existing.
        #  The index is generated given the dataset fingerprint, and should be unique.

        try:
            #self.es.indices.delete(index=index_name, ignore=[400, 404])
            response = self.es.indices.create(index=index_name)
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
        # print(response)


    def es_ingest(self, index_name: str, title: str, paragraph: str):
        """
        Ingest to ElasticSearch Index
        """
        doc = {"title": title, "text": paragraph}
        return self.es.create(index=index_name, body=doc, refresh="true", timeout=60)
        # print(response)


    def es_bulk(
        self, index_name: str, title: str, document_idx: list, passage_idx: list, document_txt: list
    ):
        actions = [
            {
                "_index": index_name,
                "_title": title,
                "_source": {
                    "document.title": title,
                    "document.idx": document_idx[i],
                    "document.passage_idx": passage_idx[i],
                    "text": document_txt[i],
                },
            }
            for i in range(len(document_txt))
        ]

        response = helpers.bulk(
            self.es, actions, chunk_size=1000, request_timeout=200, refresh="true"
        )

        return response


    def es_search_bulk(
        self, index_name: str, queries: list, k: int
    ):
        """
        Batch search in ElasticSearch Index
        """
        # todo: @MotzWanted batch search

        request = []
        req_head = [{'index': index_name}] * len(queries)
        req_body = [{"query": {"match": {"text": queries[i].lower()}},
                                "from": 0,
                                "size": k,
                                } for i in range(len(queries))]

        request = [item for sublist in zip(req_head, req_body) for item in sublist]
        
        result = self.es.msearch(body = request)
        indexes = []
        for query in result['responses']:
            temp_indexes = []
            for hit in query['hits']['hits']:
                temp_indexes.append(
                    (
                    hit['_source']['document.idx'],
                    hit['_source']['document.passage_idx'])
                    )
            indexes.append(temp_indexes)

        return indexes

    def es_search(self, index_name: str, query: str, results: int):
        # todo: @MotzWanted batch search
        response = self.es.search(
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