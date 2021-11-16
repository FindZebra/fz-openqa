import os
from functools import partial
from unittest import TestCase

import numpy as np
import pytorch_lightning as pl
import rich
import torch
from datasets import DatasetDict
from transformers import AutoTokenizer

from fz_openqa.datamodules.builders import FzCorpusBuilder, MedQaBuilder
from fz_openqa.datamodules.pipes import Predict
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.functional import get_batch_eg, cast_values_to_numpy


class Model(pl.LightningModule):
    """A simple model that takes the first 10 tokens as vector representation."""

    def forward(self, batch: Batch, **kwargs) -> Batch:
        input_ids = [v for k, v in batch.items() if k.endswith("input_ids")][0]
        return {"vector": input_ids[:, :10].float()}


class TestPredict(TestCase):
    _bert_id = "google/bert_uncased_L-2_H-128_A-2"
    n_samples = 10

    def setUp(self) -> None:
        torch.set_grad_enabled(False)
        os.environ['TOKENIZERS_PARALLELISM'] = "false"

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self._bert_id)

        # model
        self.model = Model()
        self.model.eval()
        self.model.freeze()

        # trainer
        self.trainer = pl.Trainer(checkpoint_callback=False,
                                  logger=False)

    def test_cached_predict_dataset(self):
        """Test that the output of the pipe with caching is
        the same as when using the model directly (using Dataset)."""
        dataset_builder = FzCorpusBuilder(
            tokenizer=self.tokenizer,
            use_subset=True,
            add_encoding_tokens=False,
        )
        dataset_builder.subset_size = [2]
        dataset = dataset_builder()
        dataset = dataset.select(range(self.n_samples))
        collate_fn = dataset_builder.get_collate_pipe()

        # init the predict pipe
        pipe = Predict(model=self.model, requires_cache=True)

        # cache the pipe
        pipe.cache(
            dataset,
            trainer=self.trainer,
            collate_fn=collate_fn,
            cache_dir=None,
            loader_kwargs={"batch_size": 2},
            persist=False,
        )

        # expected output
        batch = collate_fn([dataset[i] for i in range(len(dataset))])
        expected = self.model(batch)

        # process the dataset using the cache
        new_dataset = dataset.map(pipe,
                                  batched=True,
                                  with_indices=True,
                                  num_proc=2,
                                  batch_size=5)

        # compare both methods
        for i in range(len(new_dataset)):
            row = cast_values_to_numpy(new_dataset[i], dtype=np.float32)
            exp_row = cast_values_to_numpy(get_batch_eg(expected, idx=i), dtype=np.float32)
            self.assertTrue((row["vector"] == exp_row["vector"]).all())

    def test_cached_predict_dataset_dict(self):
        """Test that the output of the pipe with caching is
        the same as when using the model directly (using DatasetDict)."""
        dataset_builder = MedQaBuilder(
            tokenizer=self.tokenizer,
            use_subset=True,
            add_encoding_tokens=False,
        )
        dataset_builder.subset_size = [10, 10, 10]
        dataset = dataset_builder()
        collate_fn = dataset_builder.get_collate_pipe()

        # init the predict pipe
        pipe = Predict(model=self.model, requires_cache=True)

        # cache the pipe
        cache_paths = pipe.cache(
            dataset,
            trainer=self.trainer,
            collate_fn=collate_fn,
            cache_dir=None,
            loader_kwargs={"batch_size": 2},
            persist=False,
        )

        # check if there is a distinct cache file for each split
        self.assertIsInstance(cache_paths, dict)
        self.assertEqual(len(set(cache_paths.values())), len(dataset.keys()))

        # compute the output without using Predict
        expected = {}
        for split, dset in dataset.items():
            batch = collate_fn([dset[i] for i in range(len(dset))])
            expected[split] = self.model(batch)

        # process the dataset using the Predict+caching
        new_dataset = DatasetDict({split: dset.map(partial(pipe, split=split),
                                                   batched=True,
                                                   with_indices=True,
                                                   num_proc=2,
                                                   batch_size=5) for split, dset in
                                   dataset.items()})

        # compare both methods
        for split, dset in dataset.items():
            expected_dset = expected[split]
            new_dset = new_dataset[split]
            for i in range(len(new_dset)):
                row = cast_values_to_numpy(new_dset[i], dtype=np.float32)
                exp_row = cast_values_to_numpy(get_batch_eg(expected_dset, idx=i), dtype=np.float32)
                self.assertTrue((row["vector"] == exp_row["vector"]).all())
