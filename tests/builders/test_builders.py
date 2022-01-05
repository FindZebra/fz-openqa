import unittest
from typing import Dict, Optional, Iterable
from unittest import TestCase

from datasets import Dataset, DatasetDict, Split

from fz_openqa.datamodules.builders.corpus import MedQaCorpusBuilder, FzCorpusBuilder, \
    FZxMedQaCorpusBuilder
from fz_openqa.datamodules.builders.hf_dataset import HfDatasetBuilder
from fz_openqa.datamodules.builders.medqa import MedQaBuilder, ConcatMedQaBuilder
from fz_openqa.datamodules.builders.openqa import OpenQaBuilder
from fz_openqa.datamodules.index import ElasticSearchIndex, ElasticSearchIndexBuilder
from fz_openqa.datamodules.index.utils.es_engine import ping_es
from fz_openqa.datamodules.pipes import TextFormatter, Pipe, ExactMatch, Sampler
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer


def get_default_config():
    return {
        'cache_dir': None,
        'tokenizer': init_pretrained_tokenizer(
            pretrained_model_name_or_path="google/bert_uncased_L-2_H-128_A-2"),
        'add_encoding_tokens': True,
        'use_subset': True,
        'max_length': 512,
        'num_proc': 2,
        'text_formatter': TextFormatter(),
        'sampler': Sampler()
    }


class TestBuilder(TestCase):
    """This class loads a DatasetBuilder and tests
    1. calling it (data preprocessing)
    2. collating a batch
    # todo: check output types (tensors)
    """
    cls = HfDatasetBuilder
    config_override: Optional[Dict] = None

    def setUp(self) -> None:
        default_config = self.get_default_config()
        self.builder = self.cls(**default_config)
        self._dataset = self.builder()
        self.assertIsInstance(self._dataset, (DatasetDict, Dataset,))

    def get_default_config(self):
        default_config = get_default_config()
        if self.config_override is not None:
            default_config.update(self.config_override)

        return default_config

    @property
    def datasets(self) -> Iterable[Dataset]:
        if isinstance(self._dataset, Dataset):
            yield self._dataset
        else:
            for _, d in self._dataset.items():
                yield d

    def test_length(self):
        for dset in self.datasets:
            self.assertGreater(len(dset), 0)

    def test_columns(self):
        for dset in self.datasets:
            self.assertTrue(set(self.builder.column_names).issubset(set(dset.column_names)))

    def test_collate(self):
        collate_fn = self.builder.get_collate_pipe()
        self.assertIsInstance(collate_fn, Pipe)

        for dset in self.datasets:
            examples = [dset[i] for i in range(min(3, len(dset)))]
            batch = collate_fn(examples)
            self.assertIsInstance(batch, Dict)
            self.assertGreater(len(batch), 0)
            v = next(iter(batch.values()))
            self.assertTrue(all(len(v) == len(x) for x in batch.values()))


class TestMedQaBuilder(TestBuilder):
    config_override = {'use_subset': False}
    cls = MedQaBuilder

    def test_split_lengths(self):
        """Test the size of the splits"""
        self.assertEqual(len(self._dataset[Split.TRAIN]), 10178)
        self.assertEqual(len(self._dataset[Split.VALIDATION]), 1272)
        self.assertEqual(len(self._dataset[Split.TEST]), 1273)


class TestMedQaCorpusBuilder(TestBuilder):
    config_override = {'max_length': None, 'passage_length': 200, 'passage_stride': 200}
    cls = MedQaCorpusBuilder


class TestFzCorpusCorpusBuilder(TestMedQaCorpusBuilder):
    cls = FzCorpusBuilder


class TestFZxMedQaCorpusBuilder(TestMedQaCorpusBuilder):
    cls = FZxMedQaCorpusBuilder

@unittest.skipUnless(ping_es(), "Elastic Search is not reachable.")
class TestOpenQaBuilder(TestBuilder):
    cls = OpenQaBuilder
    dset_cls = MedQaBuilder

    def get_default_config(self):
        # dataset builder
        dataset_config = get_default_config()
        dataset_builder = self.dset_cls(**dataset_config)

        # corpus builder
        corpus_config = get_default_config()
        corpus_config.update(
            {'max_length': None, 'passage_length': 200, 'passage_stride': 200, 'use_subset': False})
        corpus_builder = MedQaCorpusBuilder(**corpus_config)

        # index builder
        return {
            'dataset_builder': dataset_builder,
            'corpus_builder': corpus_builder,
            'index_builder': ElasticSearchIndexBuilder(),
            'relevance_classifier': ExactMatch(),
            'n_retrieved_documents': 10,
            'n_documents': 5,
            'max_pos_docs': 1,
            'filter_unmatched': True,
            'num_proc': 2,
            'batch_size': 10,
        }


@unittest.skipUnless(ping_es(), "Elastic Search is not reachable.")
class TestConcatenatedOpenQaBuilder(TestBuilder):
    cls = OpenQaBuilder
    dset_cls = ConcatMedQaBuilder

    def get_default_config(self):
        # dataset builder
        dataset_config = get_default_config()
        dataset_builder = self.dset_cls(**dataset_config)

        # corpus builder
        corpus_config = get_default_config()
        corpus_config.update(
            {'max_length': None, 'passage_length': 200, 'passage_stride': 200, 'use_subset': False})
        corpus_builder = MedQaCorpusBuilder(**corpus_config)

        # index builder
        return {
            'dataset_builder': dataset_builder,
            'corpus_builder': corpus_builder,
            'index_builder': ElasticSearchIndexBuilder(),
            'relevance_classifier': ExactMatch(),
            'n_retrieved_documents': 10,
            'n_documents': 5,
            'max_pos_docs': 1,
            'filter_unmatched': True,
            'num_proc': 2,
            'batch_size': 10,
        }
