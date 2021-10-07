from .base import Pipe
from .base import Rename
from fz_openqa.utils.datastruct import Batch


class SearchCorpus(Pipe):
    """Search a Corpus object given a query"""

    def __init__(self, corpus, *, k: int):
        self.corpus = corpus
        self.k = k

    def __call__(self, query: Batch, **kwargs) -> Batch:
        result = self.corpus.search_index(query=query, k=self.k)
        result = Rename({"idx": "document.global_idx"})(result)
        query.update(**result)
        return query
