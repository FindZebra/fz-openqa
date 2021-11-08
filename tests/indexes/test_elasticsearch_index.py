import unittest

from torch import nn
from transformers import BertPreTrainedModel, AutoModel

from fz_openqa.datamodules.index import Index, ElasticSearchIndex
from fz_openqa.datamodules.index.utils.es_engine import ping_es
from fz_openqa.utils.datastruct import Batch
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


@unittest.skipIf(not ping_es(), "Elastic Search is not reachable.")
class TestElasticSearchIndex(TestIndex):
    cls: Index.__class__ = ElasticSearchIndex

    def setUp(self) -> None:
        super().setUp()
        self.model = ClsModel(AutoModel.from_pretrained(self._bert_id))
        self.model.eval()

    def test_dill_inspect(self):
        self._test_dill_inspect()

    def test_is_index(self):
        self._test_is_indexed()

    def test_search(self):
        self._test_search()
