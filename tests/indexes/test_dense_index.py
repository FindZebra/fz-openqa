import os

from datasets import Dataset
from torch import nn
from transformers import BertPreTrainedModel, AutoModel

from fz_openqa.datamodules.index import Index, FaissIndex
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


class TestDenseIndex(TestIndex):
    cls: Index.__class__ = FaissIndex

    def setUp(self) -> None:
        super().setUp()
        self.model = ClsModel(AutoModel.from_pretrained(self._bert_id))
        self.model.eval()

        # limit the number of threads for faiss
        os.environ['OMP_NUM_THREADS'] = str(2)

    @staticmethod
    def __init_index(*, corpus: Dataset, model=None, collate=None) -> Index:
        if model is None:
            model = ClsModel(AutoModel.from_pretrained(TestIndex._bert_id))
        return FaissIndex(dataset=corpus, model=model, batch_size=2, model_output_keys=['vector'],
                          collate_pipe=collate)

    def _init_index(self) -> Index:
        return self.__init_index(corpus=self.corpus, model=self.model, collate=self.corpus_collate)

    def test_dill_inspect(self):
        self._test_dill_inspect()

    def test_is_index(self):
        self._test_is_indexed()

    def test_search(self):
        self._test_search()
