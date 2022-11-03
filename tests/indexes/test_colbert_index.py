from warp_pipes import Index
from fz_openqa.datamodules.index.colbert import ColbertIndex
from tests.indexes.test_dense_index import TestDenseIndex


class TestColbertIndex(TestDenseIndex):
    """Test the faiss colbert index without using the Trainer to process data with the model"""
    cls: Index.__class__ = ColbertIndex
    _model_head = "contextual"
    index_handler = "flat"
