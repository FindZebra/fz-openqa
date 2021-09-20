import shutil
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import rich
import torch
from datasets import DatasetDict
from datasets import load_dataset
from datasets import Split
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import BatchEncoding
from transformers import PreTrainedTokenizerFast

from fz_openqa.datamodules.pipes import TokenizerPipe
from fz_openqa.datamodules.utils import HgDataset
from fz_openqa.datamodules.utils import take_subset
from fz_openqa.utils.datastruct import pprint_batch


class BaseDataModule(LightningDataModule):
    """
    A task agnostic base datamodule. This implements and showcase the
    basic functionalities of a TextDataModule.

    <original documentation>
    A DataModule implements 5 key methods:
        - prepare_data (things to do on every noe)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data
    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    # HuggingFace dataset id or local path to script
    dset_script_path_or_id = "ptb_text_only"

    # text fields from the raw datasets that should be tokenized and concatenated
    text_fields = ["sentence"]

    # name of the attributes that will be converted to tensors
    pt_attributes = ["input_ids", "attention_mask"]

    # number of data points per subset train/val/test
    subset_size = [100, 10, 10]

    def __init__(
        self,
        *,
        tokenizer: PreTrainedTokenizerFast,
        cache_dir: str = "cache/",
        train_batch_size: int = 64,
        eval_batch_size: int = 128,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        max_length: Optional[int] = 512,
        use_subset: bool = False,
        num_proc: int = 1,
        verbose: bool = True,
        train_sampler: Optional[DictConfig] = None,
        eval_sampler: Optional[DictConfig] = None,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = cache_dir
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.use_subset = use_subset
        self.num_proc = num_proc
        self.verbose = verbose

        # tokenizer and dataset
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.dataset: Optional[HgDataset] = None

        # samplers -- samplers wrap the original dataset and override the __get_item__ method
        self.train_sampler_cfg = (
            dict(train_sampler)
            if train_sampler is not None and len(train_sampler)
            else None
        )
        self.eval_sampler_cfg = (
            dict(eval_sampler)
            if eval_sampler is not None and len(eval_sampler)
            else None
        )

    def load_base_dataset(self) -> DatasetDict:
        """Load the base HuggingFace dataset."""
        return load_dataset(
            self.dset_script_path_or_id, cache_dir=self.data_dir
        )

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        self.load_base_dataset()

    def setup(self, stage: Optional[str] = None):
        """Load data and preprocess the data.
        The dataset will be stored using the attribute `self.dataset`."""
        self.dataset: HgDataset = self.load_base_dataset()
        self.dataset = self.filter_dataset(self.dataset)
        if self.use_subset:
            self.dataset = take_subset(self.dataset, self.subset_size)
        self.dataset = self.preprocess_dataset(self.dataset)

        if self.verbose:
            self.pprint()
            self.display_sample()

    def preprocess_dataset(self, dataset: HgDataset) -> HgDataset:
        """Apply processing steps to the dataset.
        Tokenization and formatting as PyTorch tensors"""
        pipe = TokenizerPipe(
            self.tokenizer,
            max_length=self.max_length,
            fields=self.text_fields,
            return_token_type_ids=False,
            add_special_tokens=False,
            return_offsets_mapping=False,
        )

        dataset = dataset.map(
            pipe,
            batched=True,
            num_proc=self.num_proc,
            desc="Tokenizing",
            remove_columns=self.text_fields,
        )
        dataset.set_format(type="torch", columns=self.pt_attributes)
        return dataset

    def filter_dataset(self, dataset: HgDataset) -> HgDataset:
        """Apply filter operation to the dataset and return"""
        return dataset

    def train_dataloader(self):
        dset = self.dataset[Split.TRAIN]
        if self.train_sampler_cfg is not None:
            dset = instantiate(self.train_sampler_cfg, dataset=dset)

        return DataLoader(
            dataset=dset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def _eval_loader(self, split):
        dset = self.dataset[split]
        if self.eval_sampler_cfg is not None:
            dset = instantiate(self.eval_sampler_cfg, dataset=dset)

        return DataLoader(
            dataset=dset,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return self._eval_loader(Split.VALIDATION)

    def test_dataloader(self):
        return self._eval_loader(Split.TEST)

    def collate_fn(
        self, batch: Any
    ) -> Union[BatchEncoding, List[Dict[str, torch.Tensor]]]:
        """The function that is used to merge examples into a batch.
        Concatenating sequences with different length requires padding them."""
        return self.tokenizer.pad(batch)

    def pprint(self):
        """Pretty print the dtaset"""
        rich.print(
            f">> Dataset: [use_subset={self.use_subset}]: \n" f"{self.dataset}"
        )

    @rank_zero_only
    def display_sample(self):
        """Sample a batch and pretty print it."""
        batch = next(iter(self.train_dataloader()))
        eval_batch = next(iter(self.val_dataloader()))
        console_width, _ = shutil.get_terminal_size()
        print(console_width * "=")
        print("=== Training Batch ===")
        print(console_width * "-")
        pprint_batch(batch)
        print("=== Validation Batch ===")
        print(console_width * "-")
        pprint_batch(eval_batch)
        print(console_width * "=")
        self.display_one_sample({k: v[0] for k, v in batch.items()})

    def display_one_sample(self, example: Dict[str, torch.Tensor]):
        """Decode and print one example from the batch"""
        console_width, _ = shutil.get_terminal_size()
        print("=== Sample ===")
        print(console_width * "-")
        rich.print(
            self.pretty_decode(example["input_ids"], skip_special_tokens=True)
        )
        print(console_width * "=")

    def pretty_decode(
        self, tokens: Union[Tensor, List[int], np.ndarray], **kwargs
    ):
        """Pretty print an encoded chunk of text"""
        n_pad_tokens = list(tokens).count(self.tokenizer.pad_token_id)
        txt = self.tokenizer.decode(tokens, **kwargs)
        return (
            f"length={len(tokens)}, padding={n_pad_tokens}, "
            f"text: [deep_sky_blue3]`{txt.replace('[PAD]', '').strip()}`"
        )
