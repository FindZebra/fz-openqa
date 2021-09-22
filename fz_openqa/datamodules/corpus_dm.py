import os
import re
from functools import partial
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Union

import rich
import torch
from datasets import concatenate_datasets
from datasets import Dataset
from datasets import DatasetDict
from datasets import load_dataset
from datasets import Split
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor

from .base_dm import BaseDataModule
from .datasets import file_corpus
from .datasets import fz_corpus
from .datasets import meqa_en_corpus
from .pipes import AddPrefix
from .pipes import Apply
from .pipes import ApplyToAll
from .pipes import Collate
from .pipes import DropKeys
from .pipes import FilterKeys
from .pipes import Forward
from .pipes import Identity
from .pipes import Lambda
from .pipes import MetaMapFilter
from .pipes import Parallel
from .pipes import Pipe
from .pipes import PrintBatch
from .pipes import ReplaceInKeys
from .pipes import SciSpacyFilter
from .pipes import Sequential
from .pipes import TokenizerPipe
from .pipes import ToNumpy
from .pipes.passage import GeneratePassages
from .utils import add_spec_token
from .utils import set_example_idx
from fz_openqa.tokenizers.static import DOC_TOKEN
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.es_functions import es_bulk
from fz_openqa.utils.es_functions import es_create_index
from fz_openqa.utils.es_functions import es_search
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

    dset_script_path_or_id = (
        file_corpus.__file__  # HuggingFace dataset id or local path to script
    )

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
        dataset = load_dataset(
            self.dset_script_path_or_id,
            cache_dir=self.data_dir,
            data_files=input_files,
        )

        if isinstance(dataset, DatasetDict):
            dataset = concatenate_datasets(list(dataset.values()))

        return dataset

    def preprocess_dataset(self, dataset: HgDataset) -> HgDataset:
        """Apply processing steps to the dataset. Tokenization and formatting as PyTorch tensors"""

        # remove title for now
        dataset = dataset.remove_columns("title")

        dataset = dataset.map(
            Sequential(
                self.get_tokenizer_pipe(), self.get_generate_passages_pipe()
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
                set_example_idx,
                batched=False,
                with_indices=True,
                desc="Indexing documents",
            )

        # casting to tensors
        dataset.set_format(
            type="torch", columns=self.pt_attributes, output_all_columns=True
        )

        return dataset

    def get_generate_passages_pipe(self):
        """Build the pipe to extract overlapping passages from the tokenized documents."""
        passage_pipe = Sequential(
            GeneratePassages(
                size=self.passage_length,
                stride=self.passage_stride,
                start_tokens=self.get_start_tokens(),
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

        tokenizer_pipe = Sequential(
            FilterKeys(lambda key: "text" in key),
            Apply(
                {"text": partial(add_spec_token, DOC_TOKEN)},
                element_wise=True,
            )
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

    def get_start_tokens(self):
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
        simple_attr_pipe = Sequential(
            Collate(keys=["idx", "document.idx", "document.passage_idx"]),
            ApplyToAll(op=lambda x: torch.tensor(x)),
        )

        # collate the questions attributes (question.input_ids, question.idx, ...)
        document_pipe = Sequential(
            Collate(keys=["document.input_ids", "document.attention_mask"]),
            ReplaceInKeys("document.", ""),
            Lambda(lambda batch: self.tokenizer.pad(batch)),
            AddPrefix("document."),
        )

        return Parallel(raw_text_pipe, simple_attr_pipe, document_pipe)

    def index(
        self,
        model: Optional[Callable] = None,
        index_mode: Optional[str] = "faiss",
        filter_mode: Optional[str] = None,
        **kwargs,
    ):
        """
        Compute vectors (sparse or dense) for the whole dataset.

        :@param model: callable that returns a vector given the batch input.
        :@param index: string that determines which index to use (faiss or bm25).
        :@param filtering: string that determines whether SciSpacy filtering is used.
        """
        if index_mode == "faiss":
            assert (
                model is not None
            ), "A model must be provided when using `faiss` indexing."
            self.build_faiss_index(model, **kwargs)

        elif index_mode == "bm25":

            # potentially filter the text field, this returns a copy of
            # `self.dataset` so it is not modified.
            dataset = self.filter_text(
                self.dataset, "document.text", filter_mode
            )

            self.build_es_index(
                dataset, index_key="idx", text_key="document.text", **kwargs
            )
        else:
            raise NotImplementedError

    def filter_text(
        self, dataset: Dataset, text_key: str, filter_mode: Optional[str]
    ) -> Dataset:
        if filter_mode is None:
            return dataset

        # select the right pipe constructor
        filter_pipe_cls = {
            "scispacy": SciSpacyFilter,
            "metamap": MetaMapFilter,
        }[filter_mode]

        # process the dataset using the filtering pipe
        return self.dataset.map(
            Sequential(
                # @idariis: added this line for debugging
                PrintBatch(header="filtering input"),
                filter_pipe_cls(text_key=text_key),
                # @idariis: added this line for debugging
                PrintBatch(header="filtering output"),
            ),
            batched=True,
            # @idariis: we need to device how to set this
            batch_size=self.eval_batch_size,
            # @idariis: potentially increase this to `self.num_proc` to use multiprocessing
            num_proc=1,
            desc="Computing corpus vectors",
        )

    def build_es_index(
        self,
        dataset: Dataset,
        *,
        index_key: str,
        text_key: str,
        verbose: bool = False,
    ):
        """Index the dataset using elastic search.
        We make sure a unique index is created for each dataset"""
        es_create_index(dataset._fingerprint)

        response = es_bulk(
            index_name=dataset._fingerprint,
            # todo: find a way to extract document titles
            title="__no_title__",
            document_idx=dataset[index_key],
            document_txt=dataset[text_key],
        )

        if verbose:
            print(get_separator("="))
            print("=== build_es_index response ===")
            print(get_separator())
            rich.print(response)
            print(get_separator("="))

    def build_faiss_index(self, model: Callable, **kwargs):
        """Index the dataset using dense vectors"""

        # compute the dense vector for the whole dataset
        self.dataset = self.dataset.map(
            self.get_batch_processing_pipe(model),
            batched=True,
            batch_size=self.eval_batch_size,
            num_proc=1,
            desc="Computing corpus vectors",
        )
        # add the dense index
        self.dataset.add_faiss_index(column=self.vectors_id, **kwargs)

    def get_batch_processing_pipe(
        self, model: Union[Callable, torch.nn.Module]
    ) -> Pipe:
        """returns a pipe that allows processing a batch of data using the model."""
        return Sequential(
            Forward(model=model, output_key=self.vectors_id),
            FilterKeys(lambda key: key == "_index_"),
            ToNumpy(),
        )

    def search_index(
        self,
        batch: Batch,
        *,
        k: int = 1,
        index_mode: Optional[str] = "faiss",
        filter_mode: Optional[str] = None,
        model: Optional[Union[Callable, torch.nn.Module]] = None,
    ) -> Batch:
        """
        Query index given a input query

        :@param query:
        :@param vector:
        :@param k: integer that sets number of results to be queried.
        :@param index: string that determines which index to use (faiss or bm25).
        :@param filtering: string that determines whether SciSpacy filtering is used.
        :@param name: string for naming index 'group'.
        """
        if index_mode == "faiss":
            # todo: this causes segmentation fault on MacOS, works fine on the cluster
            batch = self.get_batch_processing_pipe(model=model)
            vector = batch[self.vectors_id]
            _ = self.dataset.get_nearest_examples(self.vectors_id, vector, k=k)
            raise NotImplementedError

        elif index_mode == "bm25":

            # # select the right pipe constructor
            # filter_pipe_cls = {'scispacy': SciSpacyFilter,
            #                    'metamap': MetaMapFilter, }[filter_mode]
            #
            # batch = filter_pipe_cls("question.text")(batch)
            eg = {k: v[0] for k, v in batch.items()}  # todo: temporary
            return es_search(
                index_name=self.dataset._fingerprint,
                query=eg["question.text"],
                results=k,
            )

        else:
            raise NotImplementedError

    def search_index_batch(
        self,
        batch: Batch,
        k: int = 100,
        index: str = "faiss",
        **kwargs,
    ):

        """
        Query index given a batch input of queries

        :@param batch:
        """
        returned_evidence = []  # *
        for idx in range(len(batch)):  # infer batch size
            ex = {k: v[idx] for k, v in batch.items()}
            response_idx = self.search_index(
                ex["question.text"], k=k, index=index, name="corpus"
            )
            returned_evidence.append(response_idx)  # improve *

        return returned_evidence

    def exact_method(
        self,
        response,
        key: Optional[str] = None,
        queries: Optional[list] = None,
        answers: Optional[list] = None,
        answer_idxs: Optional[list] = None,
        synonyms: Optional[list] = None,
    ) -> Batch:
        """
        Compute exact matching based on whether answer is contained in document string.

        :@param batch: {
        question.text: list of N texts,
        question.input_ids:tensor of shape [N, L_q],
        answer.text: N lists of 4 texts,
        answer.input_ids: tensor of shape [N, 4, L_a],
        answer.target: tensor of shape [N,],
        answer.synonyms: N lists of M texts,
        }

        """
        out = {"version": "0.0.1", "data": []}
        discarded = {"version": "0.0.1", "data": []}

        for i, query in enumerate(queries):
            response = self.search_index(query=query, k=100, index="bm25")
            positives = []
            negatives = []
            for hit in response["hits"]:
                if answers[i][answer_idxs[i]] in hit["_source"]["text"]:
                    positives.append(hit["_source"]["text"])
                elif any(
                    synonym in hit["_source"]["text"]
                    for synonym in synonyms[i]
                ):
                    positives.append(hit["_source"]["text"])
                else:
                    negatives.append(hit["_source"]["text"])

            if positives:
                out["data"].append(
                    {
                        "question": query,
                        "answer": answers[i][0],
                        "positive": positives[0],
                        "negatives": negatives[0:10],
                    }
                )
            else:
                discarded["data"].append(
                    {
                        "question": query,
                        "answer": answers[i][0],
                        "synonyms": synonyms[i],
                        "top 10": negatives[0:10],
                    }
                )

        return out, discarded

    def query_batch(self, vectors: Tensor, k: int = 1):
        """Query the faiss index given a batch of vector queries of shape (bs, h,)"""
        vectors = vectors.cpu().numpy()
        return self.dataset.get_nearest_examples_batch(
            self.vectors_id, vectors, k=k
        )

    def pprint(self):
        """Pretty print the dtaset"""
        rich.print(
            f">> Dataset: (use_subset={self.use_subset}): \n" f"{self.dataset}"
        )


class MedQaEnDataModule(CorpusDataModule):
    dset_script_path_or_id = meqa_en_corpus.__file__


class FzCorpusDataModule(CorpusDataModule):
    dset_script_path_or_id = fz_corpus.__file__
