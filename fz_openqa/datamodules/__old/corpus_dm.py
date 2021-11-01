import os
import re
from functools import partial
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import rich
import torch
from datasets import concatenate_datasets
from datasets import Dataset
from datasets import DatasetDict
from datasets import load_dataset
from pytorch_lightning.utilities import rank_zero_only

from .base_dm import BaseDataModule
from fz_openqa.datamodules.generators import file_corpus
from fz_openqa.datamodules.generators import fz_corpus
from fz_openqa.datamodules.generators import meqa_en_corpus
from fz_openqa.datamodules.index.base import Index
from fz_openqa.datamodules.pipelines.collate import CollateAsTensor
from fz_openqa.datamodules.pipelines.collate import CollateTokens
from fz_openqa.datamodules.pipes import Apply
from fz_openqa.datamodules.pipes import Collate
from fz_openqa.datamodules.pipes import DropKeys
from fz_openqa.datamodules.pipes import FilterKeys
from fz_openqa.datamodules.pipes import Identity
from fz_openqa.datamodules.pipes import Parallel
from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.datamodules.pipes import PrintBatch
from fz_openqa.datamodules.pipes import SearchCorpus
from fz_openqa.datamodules.pipes import Sequential
from fz_openqa.datamodules.pipes import TokenizerPipe
from fz_openqa.datamodules.pipes.passage import GeneratePassages
from fz_openqa.datamodules.utils.transformations import add_spec_token
from fz_openqa.datamodules.utils.transformations import set_row_idx
from fz_openqa.tokenizers.static import DOC_TOKEN
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.pretty import get_separator
from fz_openqa.utils.pretty import pprint_batch
from fz_openqa.utils.pretty import pretty_decode

HgDataset = Union[Dataset, DatasetDict]

TXT_PATTERN = r"^.*\.txt$"


class CorpusDataModule(BaseDataModule):
    """
    A Corpus data for handling large-scale text datasets. Corpus features the following:
    * efficient caching for handling web-scale datasets (handled by `datasets`)
    * extraction of passages given a fixed length and stride
    * indexing passages
    * querying passages
    """

    dset_script_path_or_id = file_corpus.__file__  # HuggingFace dataset id or local path to script

    # name of the attributes that will be converted to
    # tensors in the preprocessing function
    pt_attributes = [
        "document.input_ids",
        "document.attention_mask",
        "document.passage_mask",
    ]

    # number of data points per subset train/val/test
    subset_size = [
        10,
    ]

    # name of the field used to store vectors
    vectors_id = "document.vectors"

    def __init__(
        self,
        *args,
        passage_length: int = 200,
        passage_stride: int = 100,
        index: Optional[Index] = None,
        add_encoding_tokens: bool = True,
        append_document_title: bool = False,
        max_length: Optional[int] = None,
        input_dir: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, max_length=max_length, **kwargs)
        assert self.max_length is None, (
            "`max_length` is not a valid argument for this dataset "
            "and should be left to None. "
            "Use the argument `passage_length` instead."
        )
        self._index = index
        self.input_dir = input_dir
        self.passage_length = passage_length
        self.passage_stride = passage_stride
        self.add_encoding_tokens = add_encoding_tokens
        if append_document_title:
            raise NotImplementedError
        self.append_document_title = append_document_title

    @classmethod
    def from_dataset(cls, corpus: Dataset):
        """Build a corpus from a loaded dataset"""
        raise NotImplementedError

    def load_base_dataset(self) -> DatasetDict:
        """Load the base HuggingFace dataset."""
        input_files = (
            [
                os.path.join(self.input_dir, p)
                for p in os.listdir(self.input_dir)
                if re.findall(TXT_PATTERN, p)
            ]
            if self.input_dir is not None
            else None
        )
        return self._load_dataset(
            self.dset_script_path_or_id,
            cache_dir=self.data_dir,
            data_files=input_files,
        )

    @staticmethod
    def _load_dataset(script, **kwargs):
        dataset = load_dataset(script, **kwargs)
        if isinstance(dataset, DatasetDict):
            dataset = concatenate_datasets(list(dataset.values()))
        return dataset

    def preprocess_dataset(self, dataset: HgDataset) -> HgDataset:
        """Apply processing steps to the dataset. Tokenization and formatting as PyTorch tensors"""

        # remove title for now
        dataset = dataset.remove_columns("title")

        dataset = dataset.map(
            Sequential(
                self.text_formatter.copy(text_key="text"),
                self.get_tokenizer_pipe(),
                self.get_generate_passages_pipe(),
            ),
            batched=True,
            num_proc=self.num_proc,
            desc="Tokenizing documents and extracting overlapping passages",
        )

        # append the prefix "document."
        for attr in dataset.column_names:
            dataset = dataset.rename_column(attr, f"document.{attr}")

        # add index column
        dataset = dataset.map(
            partial(set_row_idx, key="document.row_idx"),
            batched=False,
            num_proc=self.num_proc,
            with_indices=True,
            desc="Indexing documents",
        )

        # flatten
        dataset = dataset.flatten_indices()

        # casting to tensors
        dataset.set_format(type="torch", columns=self.pt_attributes, output_all_columns=True)

        return dataset

    def get_generate_passages_pipe(self):
        """Build the pipe to extract overlapping passages from the tokenized documents."""
        passage_pipe = Sequential(
            GeneratePassages(
                size=self.passage_length,
                stride=self.passage_stride,
                start_tokens=self.get_prefix_tokens(),
                end_tokens=[self.tokenizer.sep_token_id],
                pad_token_id=self.tokenizer.pad_token_id,
                verbose=self.verbose,
            ),
            DropKeys(["offset_mapping"]),
        )
        return passage_pipe

    def get_tokenizer_pipe(self):
        """Build a pipe to tokenize raw documents, a shortcut with the Pipe
        Parallel is added to return the original attributes as well."""

        add_spec_token_fn = partial(add_spec_token, DOC_TOKEN)
        tokenizer_pipe = Sequential(
            FilterKeys(lambda key: "text" in key),
            Apply({"text": add_spec_token_fn}, element_wise=True)
            if self.add_encoding_tokens
            else None,
            TokenizerPipe(
                self.tokenizer,
                max_length=self.max_length,
                fields=["text"],
                return_token_type_ids=False,
                add_special_tokens=False,
                return_offsets_mapping=True,
            ),
        )

        return Parallel(Identity(), tokenizer_pipe)

    def get_prefix_tokens(self):
        doc_token_id = self.tokenizer.get_vocab()[DOC_TOKEN]
        start_tokens = (
            [self.tokenizer.cls_token_id, doc_token_id]
            if self.add_encoding_tokens
            else [self.tokenizer.cls_token_id]
        )
        return start_tokens

    @rank_zero_only
    def display_sample(self):
        """Sample a batch and pretty print it."""
        batch = next(iter(self.train_dataloader()))
        print(get_separator("="))
        print("=== Corpus Batch ===")
        print(get_separator())
        pprint_batch(batch)
        print(get_separator())
        print("=== Corpus Samples ===")
        for i in range(min(3, len(list(batch.values())[0]))):
            self.display_one_sample({k: v[i] for k, v in batch.items()})
        print(get_separator("="))

    def display_one_sample(self, example: Dict[str, torch.Tensor]):
        """Decode and print one example from the batch"""
        decode_kwargs = {"skip_special_tokens": False}
        print(get_separator())
        rich.print(
            "(CORPUS) "
            + pretty_decode(
                example["document.input_ids"],
                tokenizer=self.tokenizer,
                **decode_kwargs,
            )
        )

    def get_collate_pipe(self) -> Pipe:
        """Build a Pipe to transform examples into a Batch."""

        # get the raw text questions, extract and collate
        raw_text_pipe = Collate(keys=["document.text"])

        # collate simple attributes
        simple_attr_pipe = CollateAsTensor(
            keys=[
                "idx",
                "document.idx",
                "document.passage_idx",
                "document.retrieval_score",
            ]
        )

        # collate the questions attributes (question.input_ids, question.idx, ...)
        document_pipe = CollateTokens("document.", tokenizer=self.tokenizer)

        return Parallel(raw_text_pipe, simple_attr_pipe, document_pipe)

    def build_index(
        self,
        model: Optional[Callable] = None,
        **kwargs,
    ):
        """
        Compute vectors (sparse or dense) for the whole dataset.

        :@param model: callable that returns a vector given the batch input.
        :@param index: string that determines which index to use (faiss or bm25).
        :@param filtering: string that determines whether SciSpacy filtering is used.
        """
        self._index.build(self.dataset, model=model, **kwargs)
        return self._index

    def search_index(
        self,
        query: Batch,
        *,
        k: int = 1,
        model: Optional[Union[Callable, torch.nn.Module]] = None,
        simple_collate: bool = False,
        **kwargs,
    ) -> Union[Batch, List[Dict[str, Any]]]:
        """
        Query index given a input query

        :@param query: query data stored as a Batch
        :@param k: integer that sets number of results to be queried.
        """
        return SearchCorpus(self)(query, model=model, k=k, simple_collate=simple_collate, **kwargs)


class MedQaCorpusDataModule(CorpusDataModule):
    subset_size = [1]
    dset_script_path_or_id = meqa_en_corpus.__file__


class FzCorpusDataModule(CorpusDataModule):
    dset_script_path_or_id = fz_corpus.__file__


class FZxMedQaCorpusDataModule(CorpusDataModule):
    dset_script_path_or_id: List = [
        fz_corpus.__file__,
        meqa_en_corpus.__file__,
    ]

    def load_base_dataset(self) -> DatasetDict:
        assert self.input_dir is None
        kwargs = {"cache_dir": self.data_dir}
        dsets = [self._load_dataset(s, **kwargs) for s in self.dset_script_path_or_id]
        return concatenate_datasets(dsets)