import os
import re
from collections import Counter
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import dill  # type: ignore
import rich
from datasets import concatenate_datasets
from datasets import Dataset
from datasets import DatasetDict
from datasets import load_dataset
from loguru import logger

from ..pipelines.preprocessing import FormatAndTokenize
from ..pipelines.preprocessing.text import AppendSuffix
from ..pipelines.preprocessing.text import CleanupSpecialTokens
from ..utils.transformations import set_index_column
from .adapters import DATASET_ADAPTERS
from .hf_dataset import HfDatasetBuilder
from fz_openqa.datamodules.generators import file_corpus
from fz_openqa.datamodules.generators import fz_corpus
from fz_openqa.datamodules.generators import medwiki_corpus
from fz_openqa.datamodules.generators import meqa_en_corpus
from fz_openqa.datamodules.generators import quality
from fz_openqa.datamodules.pipelines import collate
from fz_openqa.datamodules.pipelines.collate import CollateTokens
from fz_openqa.datamodules.pipes import AddPrefix
from fz_openqa.datamodules.pipes import Collate
from fz_openqa.datamodules.pipes import DropKeys
from fz_openqa.datamodules.pipes import Gate
from fz_openqa.datamodules.pipes import GeneratePassages
from fz_openqa.datamodules.pipes import Parallel
from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.datamodules.pipes import Sequential
from fz_openqa.datamodules.pipes.control.condition import In
from fz_openqa.datamodules.pipes.sentence import GenerateSentences
from fz_openqa.datamodules.utils.typing import HfDataset
from fz_openqa.tokenizers.static import DOC_TOKEN
from fz_openqa.utils.pretty import pretty_decode

TXT_PATTERN = r"^.*\.txt$"

CORPUS_GENERATORS = {
    "medqa": (meqa_en_corpus.__file__,),
    "medwiki": (medwiki_corpus.__file__, "v6"),
    "medwiki-v6": (medwiki_corpus.__file__, "v6"),
    "medwiki-v1": (medwiki_corpus.__file__, "v1"),
    "medwiki-v2": (medwiki_corpus.__file__, "v2"),
    "medwiki-v3": (medwiki_corpus.__file__, "v3"),
    "medwiki-v3-us": (medwiki_corpus.__file__, "v3-us"),
    "medwiki-v3-tw": (medwiki_corpus.__file__, "v3-tw"),
    "fz": (fz_corpus.__file__,),
    "file": (file_corpus.__file__,),
    "wikipedia": ("wikipedia", "20200501.en"),
    "quality": (quality.__file__, None),
    "race": ("race", "all"),
}


class CorpusBuilder(HfDatasetBuilder):
    """
    Builder for the Corpus Dataset.

    Attributes
    ----------
    dset_script_id
        HuggingFace dataset id or local path to script
    dset_name
        Dataset name
    pt_attributes
        name of the attributes that should be cast a Tensors
    subset_size
        number of data points per subset train/val/test
    output columns
        name of the columns
    """

    pt_attributes: List[str] = [
        "document.input_ids",
        "document.attention_mask",
    ]
    column_names = [
        "document.text",
        "document.input_ids",
        "document.attention_mask",
        "document.passage_idx",
        "document.row_idx",
        "document.idx",
        "document.question_idx",
    ]

    def __init__(
        self,
        dset_name: str = "file",
        passage_length: int = 200,
        passage_stride: int = 100,
        to_sentences: bool = False,
        input_dir: Optional[str] = None,
        append_document_title: bool = False,
        max_length: Optional[int] = None,
        **kwargs,
    ):
        super(CorpusBuilder, self).__init__(max_length=max_length, **kwargs)

        for dn in dset_name.split("+"):
            if dn not in CORPUS_GENERATORS:
                raise ValueError(
                    f"Unknown corpus {dn}, available: {list(CORPUS_GENERATORS.keys())}"
                )

        if self.max_length is not None:
            raise ValueError(
                "`max_length` is not a valid argument for this dataset "
                "and should be left to None. "
                "Use the argument `passage_length` instead."
            )

        self.dset_name = dset_name
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

        self.append_document_title = append_document_title

    @staticmethod
    def _load_dataset(*args, input_dir=None, **kwargs):
        args = list(args)

        if args[0] == "file":
            kwargs["data_files"] = [
                os.path.join(input_dir, p)
                for p in os.listdir(input_dir)
                if re.findall(TXT_PATTERN, p)
            ]

        return load_dataset(*args, **kwargs)

    def load_base_dataset(self) -> DatasetDict:
        kwargs = {"cache_dir": self.cache_dir, "input_dir": self.input_dir}

        # split dataset names using `+`
        dset_names = sorted(self.dset_name.split("+"))

        # load datasets
        dsets = []
        for dn in dset_names:
            dset_args = CORPUS_GENERATORS[dn]
            # load dataset
            dset = self._load_dataset(*dset_args, **kwargs)

            # adapt dataset
            if dn in DATASET_ADAPTERS:
                adapter = DATASET_ADAPTERS[dn]()
                _, dset = adapter(dset, num_proc=self.num_proc)

            if isinstance(dset, DatasetDict):
                dset = concatenate_datasets(list(dset.values()))
            dsets.append(dset)

        # concatenate datasets
        if len(dset_names) == 1:
            return dsets[0]
        else:
            shared_columns = set.intersection(*[set(dset.column_names) for dset in dsets])
            if any(shared_columns != set(dset.column_names) for dset in dsets):

                def drop_cols(dset: Dataset):
                    cols = set(dset.column_names)
                    cols_to_drop = cols - shared_columns
                    logger.warning(f"Dropping columns {cols_to_drop} from dataset")
                    return dset.remove_columns(list(cols_to_drop))

                dsets = [drop_cols(dset) for dset in dsets]
            return concatenate_datasets(dsets)

    def filter_dataset(self, dataset: HfDataset) -> HfDataset:
        """Apply filter operation to the dataset and return"""
        return dataset

    def preprocess_dataset(self, dataset: HfDataset) -> HfDataset:
        """Apply processing steps to the dataset. Tokenization and formatting as PyTorch tensors"""

        for attr in dataset.column_names:
            if "document." in attr:
                new_attr = attr.replace("document.", "")
                dataset = dataset.rename_column(attr, new_attr)

        # add the document index column if not already provided
        if "uid" in dataset.column_names:
            ids = dataset["uid"]
            if len(ids) != len(set(ids)):
                rich.print(Counter(ids))
                raise ValueError("Duplicated `document.uid` found in dataset")

        if "idx" not in dataset.column_names:
            dataset = set_index_column(dataset, key="idx")

        # if titles are empty, deactivate the append_document_title
        if "title" in dataset.column_names and self.append_document_title is True:
            if all(len(t) == 0 for t in dataset[:100]["title"]):
                self.append_document_title = False
                logger.warning("No title found in dataset, `append_document_title` is set to False")

        # define the pipe used for preprocessing
        preprocessing = Sequential(
            # yield sentences from each document
            Gate(self.to_sentences, self.get_generate_sentences_pipe(), update=True),
            # tokenize, only add special tokens if sentence mode is on
            Parallel(
                self.get_text_tokenizer_pipe(),
                Gate(self.append_document_title, self.get_title_tokenizer_pipe()),
            ),
            # if not sentence mode, generate equal length-passages and add the special
            # tokens to each passage,
            Gate(
                not self.to_sentences,
                self.get_generate_passages_pipe(),
                update=True,
            ),
            Gate(
                self.append_document_title,
                DropKeys(
                    keys=[
                        "title.attention_mask",
                        "title.idx",
                        "title.input_ids",
                        "title.offset_mapping",
                        "title.text",
                        "title.title",
                    ]
                ),
                update=True,
            ),
            # cleanup remaining special tokens in the text
            CleanupSpecialTokens("document.text", self.tokenizer, update=True),
        )

        # process the whole dataset (tokenization + passage generation)
        dataset = dataset.map(
            preprocessing,
            batched=True,
            batch_size=10,
            num_proc=self.num_proc,
            remove_columns=["idx", "text", "title"],
            desc="Tokenizing documents and extracting overlapping passages",
        )

        # add index column
        dataset = set_index_column(dataset, key="document.row_idx")

        return dataset

    def get_generate_sentences_pipe(self):
        return GenerateSentences(global_keys=["idx", "uid", "cui", "title", "question_idx"])

    def get_generate_passages_pipe(self):
        """Build the pipe to extract overlapping passages from the tokenized documents."""
        return GeneratePassages(
            size=self.passage_length,
            stride=self.passage_stride,
            append_document_titles=self.append_document_title,
            start_tokens=self.get_prefix_tokens(),
            end_tokens=self.get_suffix_tokens(),
            pad_token_id=self.tokenizer.pad_token_id,
            global_keys=["document.idx", "document.uid", "document.title", "document.question_idx"],
            verbose=self.verbose,
        )

    def get_text_tokenizer_pipe(self):
        """Build a pipe to tokenize raw documents, special and encoding tokens
        are added only in `to_sentence` mode."""
        add_qad_tokens = self.to_sentences and self.add_qad_tokens
        add_special_tokens = self.to_sentences and self.add_special_tokens
        return Sequential(
            FormatAndTokenize(
                prefix=None,
                key="text",
                text_formatter=None,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                add_special_tokens=add_special_tokens,
                add_qad_tokens=add_qad_tokens,
                return_offsets_mapping=True,
                qad_tokens=DOC_TOKEN,
                shape=None,
                update=True,
                input_filter=In(["text"]),
            ),
            AddPrefix("document."),
        )

    def get_title_tokenizer_pipe(self):
        """Build a pipe to tokenize raw documents, special and encoding tokens
        are added only in `to_sentence` mode."""
        return Sequential(
            AppendSuffix(text_fields="title", update=True),
            FormatAndTokenize(
                prefix=None,
                key="title",
                text_formatter=None,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                add_special_tokens=False,
                add_qad_tokens=False,
                return_offsets_mapping=True,
                qad_tokens=None,
                shape=None,
                update=True,
                input_filter=In(["title"]),
            ),
            AddPrefix("title."),
        )

    def get_prefix_tokens(self):
        """Get the prefix tokens for each passage"""
        start_token = self.tokenizer.cls_token_id
        if start_token is None:
            if self.tokenizer.bos_token != "<|endoftext|>":
                start_token = self.tokenizer.bos_token_id

        if self.add_qad_tokens:
            doc_token_id = self.tokenizer.get_vocab()[DOC_TOKEN]
            start_tokens = [start_token, doc_token_id]
        else:
            start_tokens = [start_token]

        return [s for s in start_tokens if s is not None]

    def get_suffix_tokens(self):
        """Get the suffix tokens for each passage"""
        suffix_tokens = []
        if self.add_special_tokens:
            end_token = self.tokenizer.sep_token_id
            if end_token is None:
                if self.tokenizer.eos_token != "<|endoftext|>":
                    end_token = self.tokenizer.eos_token_id

            if end_token is not None:
                suffix_tokens = [end_token]

        return suffix_tokens

    def _get_collate_pipe(self) -> Pipe:
        """Build a Pipe to transform examples into a Batch."""

        # get the raw text questions, extract and collate
        raw_collate_pipe = Collate(
            keys=["document.text", "document.title", "document.question_idx"]
        )

        # collate simple attributes
        simple_attr_pipe = collate.CollateAsTensor(
            keys=[
                "document.row_idx",
                "document.idx",
                "document.uid",
                "document.passage_idx",
                "document.proposal_score",
            ]
        )

        # collate the questions attributes (question.input_ids, question.idx, ...)
        document_pipe = CollateTokens("document.", tokenizer=self.tokenizer)

        return Parallel(raw_collate_pipe, simple_attr_pipe, document_pipe)

    def format_row(self, row: Dict[str, Any], **kwargs) -> str:
        """Decode and print one example from the batch

        Parameters
        ----------
        **kwargs
        """
        decode_kwargs = {"skip_special_tokens": False}
        return pretty_decode(
            row["document.input_ids"],
            tokenizer=self.tokenizer,
            **decode_kwargs,
        )

    @staticmethod
    def append_dot(dataset: DatasetDict) -> DatasetDict:
        """Append a dot to each title before tokenizing"""

        def add_dot(row):
            row["title"] = row["title"] + "."
            return row

        dataset.map(add_dot)
        return dataset
