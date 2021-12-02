from fz_openqa.datamodules.index import Index
from fz_openqa.datamodules.index.colbert import ColbertIndex
from tests.indexes.test_dense_index import TestFaissIndex


class TestColbertIndex(TestFaissIndex):
    """Test the faiss colbert index without using the Trainer to process data with the model"""
    cls: Index.__class__ = ColbertIndex
    _model_head = "contextual"
