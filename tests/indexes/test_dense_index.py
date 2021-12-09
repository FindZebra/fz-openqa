import os
import shutil
import tempfile
import unittest

import faiss
from datasets import Dataset
from pytorch_lightning import Trainer, LightningModule
from transformers import BertPreTrainedModel, AutoModel

from fz_openqa.datamodules.index import Index, FaissIndex
from fz_openqa.modeling.zero_shot import ZeroShot
from fz_openqa.utils.datastruct import Batch
from tests.indexes.test_base_index import TestIndex


class TestFaissIndex(TestIndex):
    """Test the faiss index without using the Trainer to process data with the model"""
    cls: Index.__class__ = FaissIndex
    _model_head = "flat"

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
    def _init_index_with(index_cls: Index.__class__, *, corpus: Dataset, model=None, collate=None, **kwargs) -> Index:
        return index_cls(dataset=corpus, model=model, batch_size=2,
                          model_output_keys=['_hq_', '_hd_'],
                          collate_pipe=collate,
                          faiss_args={
                              "factory": "Flat",
                              "metric_type": faiss.METRIC_INNER_PRODUCT},
                          persist_cache=False,
                          **kwargs)

    def _init_index(self) -> Index:
        return TestFaissIndex._init_index_with(self.cls, corpus=self.corpus, model=self.model,
                                     collate=self.corpus_collate, cache_dir=self.cache_dir)

    def test_dill_inspect(self):
        self._test_dill_inspect()

    def test_is_index(self):
        self._test_is_indexed()

    def test_search(self):
        self._test_search()
