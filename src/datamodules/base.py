import torch
from datasets import load_dataset, DatasetDict
from pytorch_lightning import LightningDataModule
from rich import print
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast, BatchEncoding
from typing import *


class BaseDatamodule(LightningDataModule):
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

    dset_id = "ptb_text_only"  # HuggingFace dataset id
    text_fields = ["sentence"]  # text fields that should be tokenized
    split_id = ["train", "validation", "test"]  # split names
    pt_attributes = ["input_ids", "attention_mask"]  # attributes to be converted into Tensors

    def __init__(
            self,
            *,
            tokenizer: PreTrainedTokenizerFast,
            data_dir: str = "data/",
            train_batch_size: int = 64,
            eval_batch_size: int = 128,
            num_workers: int = 0,
            pin_memory: bool = False,
            max_length: Optional[int] = 512,
            use_subset: bool = False,
            **kwargs,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.use_subset = use_subset

        self.max_length = max_length
        self.tokenizer = tokenizer

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def encode(self, examples: Dict[str, Any]) -> BatchEncoding:
        return self.tokenizer(*(examples[field] for field in self.text_fields))

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        self.load_datasets()

    def load_datasets(
            self,
    ) -> DatasetDict:
        if self.use_subset:
            return DatasetDict(
                {
                    split: load_dataset(
                        self.dset_id, cache_dir=self.data_dir, split=f"{split}[:{n}]"
                    )
                    for split, n in zip(
                    self.split_ids, [1000, 100, 100]
                )
                }
            )
        else:
            return load_dataset(self.dset_id, cache_dir=self.data_dir)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        dsets = self.load_datasets()
        # tokenize and format as PyTorch tensors
        dsets = dsets.map(self.encode, batched=True)
        dsets.set_format(type="torch", columns=self.pt_attributes)

        # assign splits
        self.data_train = dsets[self.split_id[0]]
        self.data_val = dsets[self.split_id[1]]
        self.data_test = dsets[self.split_id[2]]

        self.pprint()

    def pprint(self):
        print(f">> Dataset: [use_subset={self.use_subset}]: "
              f"{self.data_train.num_rows} train. rows, "
              f"{self.data_val.num_rows} val. rows, "
              f"{self.data_test.num_rows} test rows ")
        print(f">> Features:")
        l = max(map(len, self.data_train.features)) + 1
        for n, f in self.data_train.features.items():
            print(f"    - {n:{l}}: {f}")

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=self.collate_fn,
        )


    def collate_fn(self, batch: Any) -> Union[BatchEncoding, Dict[str, torch.Tensor]]:
        return self.tokenizer.pad(batch)
