import os
import re
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import dill  # type: ignore
from datasets import concatenate_datasets
from datasets import Dataset
from datasets import DatasetDict
from datasets import load_dataset
from loguru import logger
from warp_pipes import AddPrefix
from warp_pipes import CollateField
from warp_pipes import DropKeys
from warp_pipes import Gate
from warp_pipes import GeneratePassages
from warp_pipes import HfDataset
from warp_pipes import Parallel
from warp_pipes import Pipe
from warp_pipes import Sequential
from warp_pipes.core.condition import HasPrefix
from warp_pipes.core.condition import In
from warp_pipes.support.datasets_utils import get_column_names
from warp_pipes.support.pretty import pretty_decode

from ..pipelines.preprocessing import FormatAndTokenize
from ..pipelines.preprocessing.text import AppendPrefixSuffix
from ..pipelines.preprocessing.text import CleanupSpecialTokens
from ..utils.transformations import set_index_column
from .adapters import DATASET_ADAPTERS
from .hf_dataset import HfDatasetBuilder
from fz_openqa.datamodules.generators import file_corpus
from fz_openqa.datamodules.generators import fz_corpus
from fz_openqa.datamodules.generators import medwiki_corpus
from fz_openqa.datamodules.generators import meqa_en_corpus
from fz_openqa.datamodules.generators import quality
from fz_openqa.datamodules.pipes.sentence import GenerateSentences
from fz_openqa.tokenizers.static import DOC_TOKEN

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
    "findzebra": (fz_corpus.__file__,),
    "file": (file_corpus.__file__,),
    "wikipedia": ("wikipedia", "20200501.en"),
    "quality": (quality.__file__, None),
    "race": ("race", "all"),
}

DEFAULT_COLUMNS = ["text", "title"]


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
        self.append_document_title = append_document_title
        self.global_keys = None  # to be set automatically by the builder

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

            # adapt the dataset to the fz-openqa format
            if dn in DATASET_ADAPTERS:
                adapter = DATASET_ADAPTERS[dn]()
                _, dset = adapter(dset, num_proc=self.num_proc)

            if isinstance(dset, DatasetDict):
                dset = concatenate_datasets(list(dset.values()))
            dsets.append(dset)

        # concatenate datasets
        if len(dset_names) == 1:
            dataset = dsets[0]
        else:
            shared_columns = set.intersection(*[set(dset.column_names) for dset in dsets])
            if any(shared_columns != set(dset.column_names) for dset in dsets):

                def drop_cols(dset: Dataset):
                    cols = set(dset.column_names)
                    cols_to_drop = cols - shared_columns
                    if len(cols_to_drop) > 0:
                        logger.warning(f"Dropping columns {cols_to_drop} from dataset")
                        dset = dset.remove_columns(list(cols_to_drop))
                    return dset

                dsets = [drop_cols(dset) for dset in dsets]
            dataset = concatenate_datasets(dsets)

        # infer the global keys
        columns = get_column_names(dataset)
        self.global_keys = [k for k in columns if k not in DEFAULT_COLUMNS]

        return dataset

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
                raise ValueError("Duplicate `document.uid` found in dataset")

        if "idx" not in dataset.column_names:
            dataset = set_index_column(dataset, key="idx")

        # if titles are empty, deactivate the append_document_title
        if "title" in dataset.column_names and self.append_document_title is True:
            if all(len(t) == 0 for t in dataset[:100]["title"]):
                self.append_document_title = False
                logger.info("No title found in dataset, `append_document_title` is set to False")

        # define the pipe used for preprocessing
        preprocessing = Sequential(
            # tokenize the document texts and titles
            Parallel(
                self.get_text_tokenizer_pipe(),
                Gate(self.append_document_title, self.get_title_tokenizer_pipe()),
            ),
            # generate equal length-passages and add the special tokens to each passage
            self.get_generate_passages_pipe(),
            # drop the columns with `title.` prefix
            Gate(
                self.append_document_title,
                DropKeys(condition=HasPrefix("title.")),
                update=True,
            ),
            # cleanup remaining special tokens in the text
            CleanupSpecialTokens("document.text", self.tokenizer, update=True),
        )

        # process the whole dataset (tokenization + passage generation)
        cols_to_remove = get_column_names(dataset)
        dataset = preprocessing(
            dataset,
            num_proc=self.num_proc,
            batch_size=10,
            remove_columns=cols_to_remove,
            desc="Tokenizing documents and extracting overlapping passages",
        )

        # add index column
        dataset = set_index_column(dataset, key="document.row_idx")

        return dataset

    def get_generate_sentences_pipe(self):
        return GenerateSentences(global_keys=self.global_keys)

    def get_generate_passages_pipe(self):
        """Build the pipe to extract overlapping passages from the tokenized documents."""
        return GeneratePassages(
            field="document",
            size=self.passage_length,
            stride=self.passage_stride,
            start_tokens=self.get_prefix_tokens(),
            end_tokens=self.get_suffix_tokens(),
            pad_token_id=self.tokenizer.pad_token_id,
            global_keys=[k.replace("document.", "") for k in self.global_keys],
            verbose=self.verbose,
            prepend_field="title",
        )

    def get_text_tokenizer_pipe(self):
        """Build a pipe to tokenize raw documents, special and encoding tokens
        are added only in `to_sentence` mode."""
        return Sequential(
            FormatAndTokenize(
                prefix=None,
                key="text",
                text_formatter=None,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                add_special_tokens=False,
                add_qad_tokens=False,
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
            AppendPrefixSuffix(text_fields="title", suffix=". ", prefix=None, update=True),
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
        return CollateField(
            "document",
            tokenizer=self.tokenizer,
            to_tensor=["row_idx", "idx", "passage_idx", "passage_mask"],
            id="collate-documents",
        )

    def format_row(self, row: Dict[str, Any], **kwargs) -> str:
        decode_kwargs = {"skip_special_tokens": False}
        return pretty_decode(
            row["document.input_ids"],
            tokenizer=self.tokenizer,
            **decode_kwargs,
        )
