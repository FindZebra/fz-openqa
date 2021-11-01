import logging
import os
import re
from functools import partial
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import dill  # type: ignore
from datasets import concatenate_datasets
from datasets import DatasetDict
from datasets import load_dataset

from ..pipes.sentence import GenerateSentences
from .hf_dataset import HfDatasetBuilder
from fz_openqa.datamodules.generators import file_corpus
from fz_openqa.datamodules.generators import fz_corpus
from fz_openqa.datamodules.generators import meqa_en_corpus
from fz_openqa.datamodules.pipelines import collate
from fz_openqa.datamodules.pipelines.collate import CollateTokens
from fz_openqa.datamodules.pipes import Apply
from fz_openqa.datamodules.pipes import Collate
from fz_openqa.datamodules.pipes import DropKeys
from fz_openqa.datamodules.pipes import FilterKeys
from fz_openqa.datamodules.pipes import Gate
from fz_openqa.datamodules.pipes import GeneratePassages
from fz_openqa.datamodules.pipes import Identity
from fz_openqa.datamodules.pipes import Parallel
from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.datamodules.pipes import Sequential
from fz_openqa.datamodules.pipes import TokenizerPipe
from fz_openqa.datamodules.pipes import UpdateWith
from fz_openqa.datamodules.utils.transformations import add_spec_token
from fz_openqa.datamodules.utils.transformations import set_row_idx
from fz_openqa.datamodules.utils.typing import HfDataset
from fz_openqa.tokenizers.static import DOC_TOKEN
from fz_openqa.utils.pretty import pretty_decode

logger = logging.getLogger(__name__)

TXT_PATTERN = r"^.*\.txt$"


class CorpusBuilder(HfDatasetBuilder):
    # HuggingFace dataset id or local path to script
    dset_script_path_or_id = file_corpus.__file__

    # name of the attributes that will be converted to
    # tensors in the preprocessing function
    pt_attributes = [
        "document.input_ids",
        "document.attention_mask",
        "document.passage_mask",
    ]

    # number of data points per subset train/val/test
    subset_size = [3]

    # output columns
    column_names = [
        "document.text",
        "document.input_ids",
        "document.attention_mask",
        "document.passage_idx",
        "document.row_idx",
        "document.idx",
    ]

    def __init__(
        self,
        passage_length: int = 200,
        passage_stride: int = 100,
        to_sentences: bool = False,
        input_dir: Optional[str] = None,
        append_document_title: bool = False,
        max_length: Optional[int] = None,
        **kwargs,
    ):
        super(CorpusBuilder, self).__init__(max_length=max_length, **kwargs)

        assert self.max_length is None, (
            "`max_length` is not a valid argument for this dataset "
            "and should be left to None. "
            "Use the argument `passage_length` instead."
        )

        self.input_dir = input_dir
        self.passage_length = passage_length
        self.passage_stride = passage_stride
        self.to_sentences = to_sentences
        if self.to_sentences:
            logger.warning(
                f"Argument `to_sentence` is True, `passage_length`={self.passage_length} "
                f"and `passage_stride`={self.passage_stride} will be ignored."
            )
            self.passage_length = self.passage_stride = None

        if append_document_title:
            raise NotImplementedError
        self.append_document_title = append_document_title

    @staticmethod
    def _load_dataset(script, **kwargs):
        dataset = load_dataset(script, **kwargs)
        if isinstance(dataset, DatasetDict):
            dataset = concatenate_datasets(list(dataset.values()))
        return dataset

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
            cache_dir=self.cache_dir,
            data_files=input_files,
        )

    def filter_dataset(self, dataset: HfDataset) -> HfDataset:
        """Apply filter operation to the dataset and return"""
        return dataset

    def preprocess_dataset(self, dataset: HfDataset) -> HfDataset:
        """Apply processing steps to the dataset. Tokenization and formatting as PyTorch tensors"""

        # remove title for now
        dataset = dataset.remove_columns("title")

        dataset = dataset.map(
            Sequential(
                self.text_formatter.copy(text_key="text"),
                UpdateWith(Gate(self.to_sentences, GenerateSentences())),
                self.get_tokenizer_pipe(),
                UpdateWith(
                    Gate(
                        not self.to_sentences,
                        self.get_generate_passages_pipe(),
                    )
                ),
                DropKeys(["offset_mapping"]),
            ),
            batched=True,
            batch_size=10,
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
        logger.info(f"Dataset contains {len(dataset)} documents. Flatten indices and return.")
        # flatten and return
        return dataset.flatten_indices()

    def get_generate_passages_pipe(self):
        """Build the pipe to extract overlapping passages from the tokenized documents."""
        return GeneratePassages(
            size=self.passage_length,
            stride=self.passage_stride,
            start_tokens=self.get_prefix_tokens(),
            end_tokens=[self.tokenizer.sep_token_id],
            pad_token_id=self.tokenizer.pad_token_id,
            verbose=self.verbose,
        )

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
        if self.add_encoding_tokens:
            doc_token_id = self.tokenizer.get_vocab()[DOC_TOKEN]
            start_tokens = [self.tokenizer.cls_token_id, doc_token_id]
        else:
            start_tokens = [self.tokenizer.cls_token_id]
        return start_tokens

    def get_collate_pipe(self) -> Pipe:
        """Build a Pipe to transform examples into a Batch."""

        # get the raw text questions, extract and collate
        raw_text_pipe = Collate(keys=["document.text"])

        # collate simple attributes
        simple_attr_pipe = collate.CollateAsTensor(
            keys=[
                "document.row_idx",
                "document.idx",
                "document.passage_idx",
                "document.retrieval_score",
            ]
        )

        # collate the questions attributes (question.input_ids, question.idx, ...)
        document_pipe = CollateTokens("document.", tokenizer=self.tokenizer)

        return Parallel(raw_text_pipe, simple_attr_pipe, document_pipe)

    def format_row(self, row: Dict[str, Any]) -> str:
        """Decode and print one example from the batch"""
        decode_kwargs = {"skip_special_tokens": False}
        return pretty_decode(
            row["document.input_ids"],
            tokenizer=self.tokenizer,
            **decode_kwargs,
        )


class MedQaCorpusBuilder(CorpusBuilder):
    subset_size = [1]
    dset_script_path_or_id = meqa_en_corpus.__file__


class FzCorpusCorpusBuilder(CorpusBuilder):
    subset_size = [20]
    dset_script_path_or_id = fz_corpus.__file__


class FZxMedQaCorpusBuilder(CorpusBuilder):
    subset_size = [3]
    dset_script_path_or_id: List = [
        fz_corpus.__file__,
        meqa_en_corpus.__file__,
    ]

    def load_base_dataset(self) -> DatasetDict:
        assert self.input_dir is None
        kwargs = {"cache_dir": self.cache_dir}
        dsets = [self._load_dataset(s, **kwargs) for s in self.dset_script_path_or_id]
        return concatenate_datasets(dsets)