import os
import re
from typing import Any
from typing import Dict
from typing import Optional
from typing import Union

import rich
from datasets import Dataset
from datasets import DatasetDict
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

from fz_openqa.datamodules.pipes import Lambda
from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.datamodules.pipes import Sequential
from fz_openqa.datamodules.pipes import TextFormatter
from fz_openqa.datamodules.pipes import TokenizerPipe
from fz_openqa.datamodules.utils.dataset import take_subset
from fz_openqa.datamodules.utils.typing import HgDataset
from fz_openqa.utils.pretty import pretty_decode

CAMEL2SNAKE = re.compile(r"(?<!^)(?=[A-Z])")


def to_snake_format(name: str) -> str:
    return str(CAMEL2SNAKE.sub("_", name).lower())


def cache_hf_dataset(func):
    def wrapper(self, *args, **kwargs):
        if self._cache_path is None:
            # process dataset
            dataset: HgDataset = func(self, *args, **kwargs)

            if self._cache_dir is not None:
                # get fingerprint
                if isinstance(dataset, DatasetDict):
                    fingerprint = Pipe._fingerprint(
                        {k: d._fingerprint for k, d in dataset.items()}
                    )
                else:
                    fingerprint = dataset._fingerprint

                # save dataset
                self._cache_path = os.path.join(self._cache_dir, fingerprint)
                self._cache_type = type(dataset)
                if not os.path.exists(self._cache_path):
                    dataset.save_to_disk(self._cache_path)
        else:
            dataset: HgDataset = self._cache_type.load_from_disk(
                self._cache_path
            )

        return dataset

    return wrapper


class DatasetBuilder:
    _cache_path: Optional[str] = None
    _cache_type: Optional[
        Union[Dataset.__class__, DatasetDict.__class__]
    ] = None
    _cache_dir = None

    def __init__(self, *, cache_dir: Optional[str]):
        if cache_dir is None:
            self._cache_dir = None
        else:
            self._cache_dir = os.path.join(
                cache_dir, to_snake_format(self.__class__.__name__)
            )
            if not os.path.exists(self._cache_dir):
                os.makedirs(self._cache_dir)

    def __call__(self) -> Union[Dataset, DatasetDict]:
        raise NotImplementedError

    def get_collate_pipe(self) -> Pipe:
        raise NotImplementedError


class HfDatasetBuilder(DatasetBuilder):
    """This class allows loading a preprocessing a `dataset.Dataset`"""

    # HuggingFace dataset id or local path to script
    dset_script_path_or_id = "ptb_text_only"

    # text field from the raw datasets that should be tokenized
    text_field = "sentence"

    # name of the attributes that will be converted to
    # tensors in the preprocessing function
    pt_attributes = ["input_ids", "attention_mask"]

    # number of data points per subset train/val/test
    subset_size = [100, 10, 10]

    # output columns
    column_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        *,
        tokenizer: PreTrainedTokenizerFast,
        add_encoding_tokens: bool = True,
        cache_dir: str = "cache/",
        max_length: Optional[int] = 512,
        use_subset: bool = False,
        num_proc: int = 1,
        verbose: bool = False,
        text_formatter: Optional[TextFormatter] = TextFormatter(),
        **kwargs,
    ):
        super().__init__(cache_dir=cache_dir)

        self.data_dir = cache_dir
        self.use_subset = use_subset
        self.num_proc = num_proc
        self.verbose = verbose

        # tokenizer and dataset
        self.text_formatter = text_formatter
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.add_encoding_tokens = add_encoding_tokens

    @cache_hf_dataset
    def __call__(self) -> HgDataset:
        # load the dataset, potentially filter, preprocess and return
        dataset = self.load_and_filter_dataset()
        return self.preprocess_dataset(dataset)

    def load_and_filter_dataset(self) -> HgDataset:
        dataset: HgDataset = self.load_base_dataset()
        dataset = self.filter_dataset(dataset)
        if self.use_subset:
            dataset = take_subset(dataset, self.subset_size)
        return dataset

    def load_base_dataset(self) -> DatasetDict:
        """Load the base HuggingFace dataset."""
        return load_dataset(
            self.dset_script_path_or_id, cache_dir=self.data_dir
        )

    def filter_dataset(self, dataset: HgDataset) -> HgDataset:
        """Apply filter operation to the dataset and return"""
        return dataset

    def preprocess_dataset(self, dataset: HgDataset) -> HgDataset:
        """Apply processing steps to the dataset.
        Tokenization and formatting as PyTorch tensors"""
        pipe = Sequential(
            self.text_formatter.copy(text_key=self.text_field),
            TokenizerPipe(
                self.tokenizer,
                max_length=self.max_length,
                fields=self.text_field,
                return_token_type_ids=False,
                add_special_tokens=False,
                return_offsets_mapping=False,
            ),
        )

        dataset = dataset.map(
            pipe,
            batched=True,
            num_proc=self.num_proc,
            desc="Tokenizing",
            remove_columns=self.text_field,
        )
        return dataset

    def get_collate_pipe(self) -> Pipe:
        """Returns a pipe that allow collating multiple rows into one Batch"""
        return Sequential(Lambda(self.tokenizer.pad), Lambda(dict))

    def format_row(self, row: Dict[str, Any]):
        """format a row from the dataset"""
        return pretty_decode(
            row["input_ids"],
            tokenizer=self.tokenizer,
            skip_special_tokens=True,
        )
