import os
import shutil
import tempfile

from datasets import Dataset

from warp_pipes import Index, DenseIndex
from fz_openqa.modeling.zero_shot import ZeroShot
from tests.indexes.test_base_index import TestIndex


class TestDenseIndex(TestIndex):
    """Test the faiss index without using the Trainer to process data with the model"""
    cls: Index.__class__ = DenseIndex
    _model_head = "flat"
    index_handler= "flat"
    index_factory = "Flat"

    def setUp(self) -> None:
        super().setUp()
        self.model = ZeroShot(self._bert_id, head=self._model_head)
        self.model.eval()

        # limit the number of threads for faiss
        os.environ['OMP_NUM_THREADS'] = str(2)

        self.cache_dir = str(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.cache_dir)

    @staticmethod
    def _init_index_with(index_cls: Index.__class__,
                         *,
                         corpus: Dataset,
                         model=None,
                         collate=None,
                         handler=None,
                         index_factory=None,
                         **kwargs) -> Index:
        return index_cls(dataset=corpus,
                         model=model,
                         batch_size=2,
                         model_output_keys=['_hq_', '_hd_'],
                         collate_pipe=collate,
                         index_factory=index_factory,
                         handler=handler,
                         dtype="float32",
                         persist_cache=False,
                         # colbert p value
                         p=100,
                         **kwargs)

    def _init_index(self) -> Index:
        return TestDenseIndex._init_index_with(self.cls,
                                               corpus=self.corpus,
                                               model=self.model,
                                               collate=self.corpus_collate,
                                               cache_dir=self.cache_dir,
                                               handler=self.index_handler,
                                               index_factory=self.index_factory)


    def test_dill_inspect(self):
        self._test_dill_inspect()

    def test_search(self):
        self._test_search()
