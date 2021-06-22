import shutil
from functools import partial
from typing import *

import datasets
import rich
import torch
from datasets import load_dataset, DatasetDict, Split
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast, BatchEncoding

HgDataset = Union[Dataset, DatasetDict]


class BaseDataModule(LightningDataModule):
    """
    A base LightningDataModule for the PennTreeBank dataset as example.
    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))
    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data
    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    dset_script_path_or_id = (
        "ptb_text_only"  # HuggingFace dataset id or local path to script
    )
    text_fields = ["sentence"]  # text fields that should be tokenized
    split_ids = [
        datasets.Split.TRAIN,
        datasets.Split.VALIDATION,
        datasets.Split.TEST,
    ]  # split names
    pt_attributes = [
        "input_ids",
        "attention_mask",
    ]  # attributes to be converted into Tensors

    def __init__(
            self,
            *,
            tokenizer: PreTrainedTokenizerFast,
            cache_dir: str = "cache/",
            train_batch_size: int = 64,
            eval_batch_size: int = 128,
            num_workers: int = 0,
            pin_memory: bool = False,
            persistent_workers: bool = False,
            max_length: Optional[int] = 512,
            use_subset: bool = False,
            verbose: bool = True,
            corpus: Optional['BaseDataModule'] = None,
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
        self.verbose = verbose

        # corpus object
        self.corpus = corpus

        # tokenizer and dataset
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.dataset: Optional[HgDataset] = None

    def tokenize_examples(
            self,
            examples: Dict[str, List[Any]],
            *,
            tokenizer: PreTrainedTokenizerFast,
            max_length: Optional[int],
            preprocess_fn: Optional[Callable] = None,
            **kwargs
    ) -> Union[Dict, BatchEncoding]:
        """Tokenize a batch of examples and truncate if `max_length` is provided.
        The input format is:
        examples = {
            attribute_name: list of attribute values
        }
        """
        if preprocess_fn is None:
            preprocess_fn = lambda x: x

        text_fields = {field: list(map(preprocess_fn, examples[field])) for field in self.text_fields}
        return tokenizer(
            *text_fields.values(),
            max_length=max_length,
            truncation=max_length is not None,
            **kwargs,
        )

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        self.load_base_dataset()
        if self.corpus is not None:
            self.corpus.prepare_data()

    def load_base_dataset(self) -> DatasetDict:
        """Load the base HuggingFace dataset."""
        return load_dataset(self.dset_script_path_or_id, cache_dir=self.data_dir)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        self.dataset: HgDataset = self.load_base_dataset()
        self.dataset = self.filter_dataset(self.dataset)
        if self.use_subset:
            self.dataset = self.take_subset(self.dataset)
        self.dataset = self.preprocess_dataset(self.dataset)

        if self.verbose:
            self.pprint()
            self.display_sample()

        if self.corpus is not None:
            self.corpus.setup()
            if self.verbose:
                console_width, _ = shutil.get_terminal_size()
                print("=== Corpus ===")
                print(console_width * "*")
                self.corpus.pprint()
                self.corpus.display_sample()
                print(console_width * "*")

    def take_subset(self, dataset: HgDataset) -> HgDataset:
        """Take a subset of the dataset and return."""
        if isinstance(dataset, DatasetDict):
            return DatasetDict(
                {
                    k: dset.select(range(n))
                    for n, (k, dset) in zip([100, 10, 10], self.dataset.items())
                }
            )
        elif isinstance(dataset, Dataset):
            return dataset.select(range(100))
        else:
            raise NotImplementedError

    def preprocess_dataset(self, dataset: HgDataset) -> HgDataset:
        """Apply processing steps to the dataset. Tokenization and formatting as PyTorch tensors"""
        fn = partial(
            self.tokenize_examples, tokenizer=self.tokenizer, max_length=self.max_length
        )
        dataset = dataset.map(fn, batched=True)
        dataset.set_format(type="torch", columns=self.pt_attributes)
        return dataset

    def filter_dataset(self, dataset: HgDataset) -> HgDataset:
        """Apply filter operation to the dataset and return"""
        return dataset

    def pprint(self):
        """Pretty print the dtaset"""
        rich.print(
            f">> Dataset: [use_subset={self.use_subset}]: \n"
            f"{self.dataset}"
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.dataset[Split.TRAIN],
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.dataset[Split.VALIDATION],
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.dataset[Split.TEST],
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch: Any) -> Union[BatchEncoding, Dict[str, torch.Tensor]]:
        """The function that is used to merge examples into a batch.
        Concatenating sequences with different length requires padding them."""
        return self.tokenizer.pad(batch)

    @rank_zero_only
    def display_sample(self):
        """Sample a batch and pretty print it."""
        batch = next(iter(self.train_dataloader()))
        console_width, _ = shutil.get_terminal_size()
        print(console_width * "=")
        print("=== Training Batch ===")
        print(console_width * "-")
        for k, v in batch.items():
            rich.print(f"   - {k}: {v.shape} <{v.dtype}>")
        print(console_width * "=")
        self.display_one_sample({k: v[0] for k, v in batch.items()})

    def display_one_sample(self, example: Dict[str, torch.Tensor]):
        """Decode and print one example from the batch"""
        console_width, _ = shutil.get_terminal_size()
        print("=== Sample ===")
        print(console_width * "-")
        rich.print(self.tokenizer.decode(example["input_ids"], skip_special_tokens=True))
        print(console_width * "=")
