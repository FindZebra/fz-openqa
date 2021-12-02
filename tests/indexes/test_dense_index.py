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

    def setUp(self) -> None:
        super().setUp()
        self.model = ZeroShot(self._bert_id, head="flat")
        self.model.eval()

        # limit the number of threads for faiss
        os.environ['OMP_NUM_THREADS'] = str(2)

        self.cache_dir = str(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.cache_dir)

    @staticmethod
    def _init_index_with(*, corpus: Dataset, model=None, collate=None, **kwargs) -> Index:
        return FaissIndex(dataset=corpus, model=model, batch_size=2,
                          model_output_keys=['_hq_', '_hd_'],
                          collate_pipe=collate,
                          faiss_args={
                              "type": "flat",
                              "metric_type": faiss.METRIC_INNER_PRODUCT},
                          persist_cache=False,
                          **kwargs)

    def _init_index(self) -> Index:
        return self._init_index_with(corpus=self.corpus, model=self.model,
                                     collate=self.corpus_collate, cache_dir=self.cache_dir)

    def test_dill_inspect(self):
        self._test_dill_inspect()

    def test_is_index(self):
        self._test_is_indexed()

    def test_search(self):
        self._test_search()


class TestFaissIndexWithTrainer(TestFaissIndex):
    """Test the faiss index using the Trainer to accelerate processing the data with the model"""

    @staticmethod
    def _init_index_with(**kwargs) -> Index:
        trainer = Trainer(checkpoint_callback=False, logger=False)
        return TestFaissIndex._init_index_with(trainer=trainer, **kwargs)
