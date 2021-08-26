import random
from unittest import TestCase

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from fz_openqa.datamodules.fz_x_medqa_dm import FZxMedQADataModule


class TestFZxMedQaDm(TestCase):
    def setUp(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        self.dm = FZxMedQADataModule(
            tokenizer=tokenizer,
            train_batch_size=10,
            eval_batch_size=10,
            verbose=False,
            filter_gold=True
        )
        self.dm.prepare_data()
        self.dm.setup()

        # get the training set and get examples
        self.dset = self.dm.dataset['train']
        self.text = self.dm.text_data['train']
        self.examples = [self.dset[random.randint(0, len(self.dset))] for _ in range(10)]

        # build indexes
        self.dset_index = self._build_index(self.dset, columns=['question.idx'])
        self.text_index = self._build_index(self.text, columns=['question.idx'])
        print(self.text_index)

    def _build_index(self, dset, columns):
        frame = pd.DataFrame(
            dset.remove_columns(
                [
                    c
                    for c in dset.column_names
                    if c not in columns
                ]
            )
        ).astype(np.int32)
        frame['_index_'] = frame.index
        return frame

    def test_sampling_examples(self):
        """test that examples retrieved from dm.dataset correspond to the original
        data when matching over question.idx"""
        for ex in self.examples:
            question_idx = int(ex['question.idx'])
            indexes = self.text_index[self.text_index['question.idx']==question_idx]
            assert len(indexes) == 1
            idx = indexes['_index_'].values[0]
            # check consistency between the original text and encoded text
            self.assertEqual(ex['question.text'][:60], self.text['question'][idx][:60])
            self.assertEqual(ex['document.text'][:60], self.text['document'][idx][:60])
            for i, ans in enumerate(ex['answer.text']):
                self.assertEqual(ans[:60], self.text['answer'][idx][i][:60])

    def test_collate_fn(self):
        print()

        self.assertTrue(True)
