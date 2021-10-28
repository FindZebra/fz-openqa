from typing import Dict, Optional
from unittest import TestCase

from datasets import Dataset, DatasetDict

from fz_openqa.datamodules.builders.base import DatasetBuilder
from fz_openqa.datamodules.builders.corpus import MedQaCorpusBuilder, FzCorpusCorpusBuilder, \
    FZxMedQaCorpusBuilder
from fz_openqa.datamodules.builders.medqa import MedQABuilder
from fz_openqa.datamodules.pipes import TextFormatter
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer
from fz_openqa.utils.pretty import pprint_batch


def get_default_config():
    return {
        'cache_dir': None,
        'tokenizer': init_pretrained_tokenizer(
            pretrained_model_name_or_path="google/bert_uncased_L-2_H-128_A-2"),
        'add_encoding_tokens': True,
        'use_subset': True,
        'max_length': 512,
        'num_proc': 2,
        'verbose': False,
        'text_formatter': TextFormatter()
    }


class TestBuilder(TestCase):
    cls = DatasetBuilder
    config_override: Optional[Dict] = None

    def setUp(self) -> None:
        default_config = get_default_config()
        if self.config_override is not None:
            default_config.update(self.config_override)

        self.builder = self.cls(**default_config)
        self._dataset = self.builder()
        self.assertIsInstance(self._dataset, (Dataset, DatasetDict,))

    @property
    def dataset(self):
        if isinstance(self._dataset, Dataset):
            return self._dataset
        else:
            return self._dataset["train"]

    def test_length(self):
        self.assertGreater(len(self.dataset), 0)

    def test_collate(self):
        dset = self.dataset
        pipe = self.builder.get_collate_pipe()
        examples = [dset[i] for i in range(min(3, len(dset)))]
        batch = pipe(examples)
        self.assertIsInstance(batch, Dict)
        self.assertGreater(len(batch), 0)
        v = next(iter(batch.values()))
        self.assertTrue(all(len(v) == len(x) for x in batch.values()))

        pprint_batch(batch)


class TestMedQABuilder(TestBuilder):
    cls = MedQABuilder


class TestMedQaCorpusBuilder(TestBuilder):
    config_override = {'max_length': None, 'passage_length': 200, 'passage_stride': 200}
    cls = MedQaCorpusBuilder


class TestFzCorpusCorpusBuilder(TestMedQaCorpusBuilder):
    cls = FzCorpusCorpusBuilder


class TestFZxMedQaCorpusBuilder(TestMedQaCorpusBuilder):
    cls = FZxMedQaCorpusBuilder
