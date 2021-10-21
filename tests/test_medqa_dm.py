import random
import tempfile
from unittest import TestCase

import numpy as np
import pandas as pd
from datasets import Split
from transformers import AutoTokenizer

from fz_openqa.datamodules import MedQaDataModule


class TestMedQaDm(TestCase):
    def setUp(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        self.dm = MedQaDataModule(
            tokenizer=tokenizer,
            train_batch_size=10,
            eval_batch_size=10,
            use_subset=False,
            num_proc=4,
            verbose=False,
        )
        self.dm.prepare_data()
        self.dm.setup()

    def test_split_lengths(self):
        """Test the size of the splits"""

        self.assertEqual(len(self.dm.dataset[Split.TRAIN]), 10178)
        self.assertEqual(len(self.dm.dataset[Split.VALIDATION]), 1272)
        self.assertEqual(len(self.dm.dataset[Split.TEST]), 1273)

    def test_batch(self):
        batch = next(iter(self.dm.train_dataloader()))
        required_attributes = ["question.text",
                               "question.input_ids",
                               "question.attention_mask",
                               "answer.text",
                               "answer.input_ids",
                               "answer.attention_mask",
                               ]

        for key in required_attributes:
            self.assertIn(key, batch.keys())
