import rich

from fz_openqa.datamodules.index import Index
from fz_openqa.datamodules.index.colbert import ColbertIndex
from tests.indexes.test_dense_index import TestDenseIndex


class TestColbertMultiIndex(TestDenseIndex):
    """Test the faiss colbert index without using the Trainer to process data with the model"""
    cls: Index.__class__ = ColbertIndex
    _model_head = "contextual"
    index_handler = "multi"
    index_factory = "torch"


    def _supplementary_test_search(self, query:dict, data: dict):
        """make sure that retrieved passages are only in the target document."""
        scores = data["score"]
        indexes = data["index"]
        query_doc_ids = query["question.document_idx"]
        row_doc_ids = self.corpus["document.idx"]

        for k in range(len(scores)):
            doc_id_k = query_doc_ids[k]
            scores_k = scores[k]
            # drop indices with -inf score
            indexes_k = indexes[k][scores_k >= -1e8]
            retrieved_doc_idx = [row_doc_ids[i] for i in indexes_k]
            assert all(doc_id_k == doc_id for doc_id in retrieved_doc_idx)
