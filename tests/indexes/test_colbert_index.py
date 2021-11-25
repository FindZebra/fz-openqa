import os
import shutil
import tempfile
import unittest

from datasets import Dataset
from pytorch_lightning import Trainer, LightningModule
from transformers import BertPreTrainedModel, AutoModel

from fz_openqa.datamodules.index import Index
from fz_openqa.datamodules.index.colbert import ColbertIndex
from fz_openqa.utils.datastruct import Batch
from tests.indexes.test_base_index import TestIndex

class ClsModel(LightningModule):
    def __init__(self, bert: BertPreTrainedModel):
        super(ClsModel, self).__init__()
        self.bert = bert

    def forward(self, batch: Batch) -> Batch:
        if any('document' in key for key in batch.keys()):
            prefix = "document"
        elif any('question' in key for key in batch.keys()):
            prefix = 'question'
        else:
            raise ValueError("No compatible key found in batch")

        output = self.bert(batch[f'{prefix}.input_ids'], batch[f'{prefix}.attention_mask'])
        return {f'{prefix}.vector': output.last_hidden_state[:, 0]}


class TestColbertIndex(TestIndex):
    """Test the faiss colbert index without using the Trainer to process data with the model"""
    cls: Index.__class__ = ColbertIndex

    def setUp(self) -> None:
        super().setUp()
        self.model = ClsModel(AutoModel.from_pretrained(self._bert_id))
        self.model.eval()

        # limit the number of threads for faiss
        os.environ['OMP_NUM_THREADS'] = str(2)

        self.cache_dir = str(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.cache_dir)

    @staticmethod
    def _init_index_with(*, corpus: Dataset, model=None, collate=None, **kwargs) -> Index:
        if model is None:
            model = ClsModel(AutoModel.from_pretrained(TestIndex._bert_id))
        return ColbertIndex(dataset=corpus, model=model, batch_size=2,
                          model_output_keys=['question.vector', 'document.vector'],
                          collate_pipe=collate, **kwargs)

    def _init_index(self) -> Index:
        return self._init_index_with(corpus=self.corpus, model=self.model,
                                     collate=self.corpus_collate, cache_dir=self.cache_dir)

    def test_dill_inspect(self):
        self._test_dill_inspect()

    def test_is_index(self):
        self._test_is_indexed()

    def test_search(self):
        self._test_search()


class TestFaissIndexWithTrainer(TestColbertIndex):
    """Test the faiss colbert index using the Trainer to accelerate processing the data with the model"""

    @staticmethod
    def _init_index_with(**kwargs) -> Index:
        trainer = Trainer(checkpoint_callback=False, logger=False)
        return TestColbertIndex._init_index_with(trainer=trainer, **kwargs)

    @unittest.skip(f"cannot pickle trainer for now.")
    def test_dill_inspect(self):
        self._test_dill_inspect()
