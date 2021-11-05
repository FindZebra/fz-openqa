import multiprocessing as mp
from functools import partial

import rich
from datasets import Dataset
from torch import nn
from transformers import BertPreTrainedModel, AutoModel
import dill
from fz_openqa.datamodules.index import Index, FaissIndex
from fz_openqa.datamodules.pipelines.collate import CollateTokens
from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.pretty import pprint_batch
from tests.indexes.test_base_index import TestIndex


class ClsModel(nn.Module):
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
        return {'vector': output.last_hidden_state[:, 0]}


class TestDenseIndex(TestIndex):
    cls: Index.__class__ = FaissIndex

    def setUp(self) -> None:
        super(TestDenseIndex, self).setUp()
        self.model = ClsModel(AutoModel.from_pretrained(self._bert_id))
        self.model.eval()


    @staticmethod
    def _init_index(*, corpus: Dataset, k: int, model=None, collate=None) -> Index:
        if model is None:
            model = ClsModel(AutoModel.from_pretrained(TestIndex._bert_id))
        return FaissIndex(dataset=corpus, model=model, batch_size=2, k=k, model_output_keys=['vector'], collate_pipe=collate)

    def init_index(self) -> Index:
        return self._init_index(corpus=self.corpus, k=self.k, model=self.model, collate=self.corpus_collate)

    @staticmethod
    def get_index_fingerprint(*args, **kwargs) -> str:
        fingerprint =  str(Pipe._fingerprint(TestDenseIndex._init_index(**kwargs)))
        rich.print(f">>> {args}: fingerprint={fingerprint}")
        return fingerprint

    def test_fingerprint(self):
        pool = mp.Pool()
        fn = partial(self.get_index_fingerprint, corpus=self.corpus, k=self.k)
        fingerprints = pool.map(fn, [1,2,3])
        rich.print(fingerprints)

    def test_is_indexed(self):
        index = self.init_index()
        self.assertTrue(index.is_indexed)

    def test_search(self):
        assert False

    def test_get_example(self):
        assert False
