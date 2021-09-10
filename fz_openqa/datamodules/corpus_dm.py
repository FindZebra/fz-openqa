import os
import re
import shutil
from collections import defaultdict
from functools import partial
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import datasets
import numpy as np
import rich
import torch
from datasets import DatasetDict
from datasets import load_dataset
from datasets import Split
from elasticsearch import Elasticsearch
from pytorch_lightning.utilities import move_data_to_device
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor
from torch.utils.data import Dataset

from .base_dm import BaseDataModule
from .collate import collate_and_pad_attributes
from .collate import extract_and_collate_attributes_as_list
from .datasets import file_corpus
from .datasets import fz_corpus
from .datasets import meqa_en_corpus
from .passage import gen_passages
from fz_openqa.tokenizers.static import DOC_TOKEN
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.datastruct import pprint_batch
from fz_openqa.utils.es_functions import es_bulk
from fz_openqa.utils.es_functions import es_create_index

HgDataset = Union[Dataset, DatasetDict]

TXT_PATTERN = r"^.*\.txt$"

# add more fields to elasticsearch index within "properties"
es_config = {
    "settings": {
        "number_of_shards": 1,
        "analysis": {
            "analyzer": {
                "stop_standard": {
                    "type": "standard",
                    " stopwords": "_english_",
                }
            }
        },
    },
    "mappings": {
        "properties": {
            "ducment.attention_mask": {"type": "dense_vector", "dims": 2},
            "document.idx": {"type": "dense_vector", "dims": 1},
            "ducment.input_ids": {"type": "dense_vector", "dims": 2},
            "document.text": {
                "type": "text",
                "analyzer": "standard",
                "similarity": "BM25",
            },
            "docment.passage_idx": {"type": "dense_vector", "dims": 1},
            "document.passage_mask": {"type": "dense_vector", "dims": 2},
            "document.vectors": {"type": "dense_vector", "dims": 2},
        }
    },
}


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
    split_ids = [
        datasets.Split.TRAIN,
        datasets.Split.VALIDATION,
        datasets.Split.TEST,
    ]  # split names
    pt_attributes = [
        "idx",
        "passage_idx",
        "input_ids",
        "attention_mask",
        "passage_mask",
    ]  # attributes to be converted into Tensors
    vectors_id = "document.vectors"

    def __init__(
        self,
        *args,
        input_dir: Optional[str] = None,
        passage_length: int = 200,
        passage_stride: int = 200,
        passage_min_length: Optional[int] = 100,
        max_length: Optional[int] = None,
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
        self.passage_min_length = passage_min_length
        if self.append_document_title:
            # appending the title is quite complicated as it required
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
        return load_dataset(
            self.dset_script_path_or_id,
            cache_dir=self.data_dir,
            data_files=input_files,
        )

    @staticmethod
    def add_idx(example: Dict[str, Any], idx: int):
        example["idx"] = idx
        return example

    def preprocess_dataset(self, dataset: HgDataset) -> HgDataset:
        """Apply processing steps to the dataset. Tokenization and formatting as PyTorch tensors"""

        # remove title for now
        dataset.remove_columns_("title")

        # add index column
        dataset = dataset.map(
            self.add_idx, batched=False, with_indices=True, desc="Indexing"
        )

        # tokenize the text
        fn = partial(
            self.tokenize_examples,
            fields=["text"],
            output_key=None,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            return_token_type_ids=False,
            add_special_tokens=False,
            return_offsets_mapping=True,
            add_encoding_tokens=False,
        )
        dataset = dataset.map(
            fn, batched=True, num_proc=self.num_proc, desc="Tokenizing text"
        )

        # generate passages of equal size
        doc_token_id = self.tokenizer.get_vocab()[DOC_TOKEN]
        start_tokens = (
            [self.tokenizer.cls_token_id, doc_token_id]
            if self.add_encoding_tokens
            else [self.tokenizer.cls_token_id]
        )
        gen_passages = partial(
            self.generate_passages,
            size=self.passage_length,
            stride=self.passage_stride,
            start_tokens=start_tokens,
            end_tokens=[self.tokenizer.sep_token_id],
            pad_token_id=self.tokenizer.pad_token_id,
            verbose=self.verbose,
        )
        dataset = dataset.map(
            gen_passages,
            batched=True,
            num_proc=self.num_proc,
            desc="Extracting passages",
        )

        # remove passages =< passage_min_length
        if self.passage_min_length is not None:
            print(f">>> 1: dset: {len(dataset['train'])}")
            dataset = dataset.filter(
                lambda ex: sum(ex["attention_mask"]) > self.passage_min_length
            )
            print(f">>> 2: dset: {len(dataset['train'])}")

        # dropping unnecessary columns and cast into tensors
        dataset = dataset.remove_columns(["offset_mapping"])
        dataset.set_format(
            type="torch", columns=self.pt_attributes, output_all_columns=True
        )

        # append the prefix "document."
        dataset = dataset.rename_column("text", "document.text")
        for attr in [
            "input_ids",
            "attention_mask",
            "passage_mask",
            "idx",
            "passage_idx",
        ]:
            dataset = dataset.rename_column(attr, f"document.{attr}")

        return dataset

    @staticmethod
    def generate_passages(
        examples: Dict[str, List[Any]],
        *,
        size: int,
        stride: int,
        start_tokens: List[int],
        end_tokens: List[int],
        pad_token_id: int,
        verbose: bool = True,
    ) -> Dict[str, List]:

        if verbose:
            lens = list(map(len, examples["input_ids"]))
            rich.print(
                f">> @CorpusDataModule.generate_passages: Number of tokens per documents: "
                f"mean={np.mean(lens):.1f}, std={np.std(lens):.1f} [{min(lens)} - {max(lens)}]"
            )

        # generate passages for all arguments, the dictionary bellow describes the configuration of the
        # function `gen_passages` for each key.
        base_args = {"size": size, "stride": stride}
        gen_passage_args = {
            "input_ids": {
                "pad_token": pad_token_id,
                "start_tokens": start_tokens,
                "end_tokens": end_tokens,
                **base_args,
            },
            "attention_mask": {
                "pad_token": 0,
                "start_tokens": [0 for _ in start_tokens],
                "end_tokens": [0 for _ in end_tokens],
                **base_args,
            },
            "offset_mapping": {
                "pad_token": [-1, -1],
                "start_tokens": [[-1, -1] for _ in start_tokens],
                "end_tokens": [[-1, -1] for _ in end_tokens],
                **base_args,
            },
        }

        indexes, output = CorpusDataModule.generate_passages_for_all_keys(
            examples,
            keys=["input_ids", "attention_mask", "offset_mapping"],
            args=gen_passage_args,
        )

        # extract document.text
        output["text"] = [
            CorpusDataModule.extract_passage_text_from_doc(
                examples["text"][idx], ofs_ids
            )
            for idx, ofs_ids in zip(indexes, output["offset_mapping"])
        ]

        # drop unnecessary attributes
        for k in ["offset_mapping"]:
            output[k] = [None for _ in output["input_ids"]]

        return output

    @staticmethod
    def generate_passages_for_all_keys(
        examples: Dict[str, List[Any]],
        keys: List[str],
        args: Dict[str, Dict[str, Any]],
    ) -> Tuple[List[int], Batch]:
        """
        This functions generate the passages for each attribute in `keys`, the `arg` dictionary
        must contain an entry for all `keys`. The first pass is used to store the document/example indexes
        and compute the `passage_mask`.

        The passage mask is used for segmentation, and is optional for this project.
        In this context, all tokens are attributed to a single passage,
        although they appear in multiple passages (strides).
        The passage mask indicates if a token is attributed to this specific passage.

        return:
            - indexes: index of the parent example for each passage
            - output: Batch of data for all keys + `idx` (document id) and `passage_mask`
        """
        assert "idx" in examples.keys()
        assert all(key in args.keys() for key in keys)
        L = len(next(iter(examples.values())))
        assert all(L == len(x) for x in examples.values())

        first_key, *other_keys = keys
        output = defaultdict(list)
        indexes = []
        for idx, (doc_idx, example) in enumerate(
            zip(examples["idx"], examples[first_key])
        ):

            # do a first pass to compute the passage masks
            for pas_idx, (passage, passage_mask) in enumerate(
                gen_passages(example, **args[first_key], return_mask=True)
            ):
                indexes += [idx]
                output["idx"].append(doc_idx)
                output["passage_idx"].append(pas_idx)
                output["passage_mask"].append(passage_mask)
                output[first_key].append(passage)

            # do another pass to generate the passages for each remaining attribute
        for key in other_keys:
            for example in examples[key]:
                passages = gen_passages(
                    example, **args[key], return_mask=False
                )
                for i, passage in enumerate(passages):
                    output[key].append(passage)

        # check output consistency and return
        L = len(list(next(iter(output.values()))))
        assert all(len(v) == L for v in output.values())
        return indexes, output

    @staticmethod
    def extract_passage_text_from_doc(
        document: str, offset_mapping: List[Tuple[int, int]]
    ) -> str:
        """
        Extract the text passage from the original document
        given the offset mapping of the passage
        """
        indexes = [
            x for idxes_tok in offset_mapping for x in idxes_tok if x >= 0
        ]
        return document[min(indexes) : max(indexes)]

    @rank_zero_only
    def display_sample(self):
        """Sample a batch and pretty print it."""
        batch = next(iter(self.train_dataloader()))
        console_width, _ = shutil.get_terminal_size()
        print(console_width * "=")
        print("=== Corpus Batch ===")
        print(console_width * "-")
        pprint_batch(batch)
        print(console_width * "=")
        print("=== Corpus Samples ===")
        for i in range(min(3, len(list(batch.values())[0]))):
            self.display_one_sample({k: v[i] for k, v in batch.items()})
        print(console_width * "=")

    def display_one_sample(self, example: Dict[str, torch.Tensor]):
        """Decode and print one example from the batch"""
        console_width, _ = shutil.get_terminal_size()
        decode_kwargs = {"skip_special_tokens": False}
        print(console_width * "-")
        rich.print(
            "(CORPUS) "
            + self.repr_ex(example, "document.input_ids", **decode_kwargs)
        )

    def collate_fn(self, examples: List[Dict[str, Any]]) -> Batch:
        """The function that is used to merge examples into a batch.
        Concatenating sequences with different length requires padding them."""

        # get the raw text inputs, extract and collate
        examples, text_outputs = extract_and_collate_attributes_as_list(
            examples, attribute="text", key="document"
        )

        # collate the tensor attributes: input_ids, idx, ...
        tensor_outputs = collate_and_pad_attributes(
            examples, tokenizer=self.tokenizer, key="document", exclude="text"
        )

        return {**tensor_outputs, **text_outputs}

    def truncate_examples_to_max_length(self, output, *, key: str):
        # infer `max_length`
        tokens = [t for t in output[f"{key}.input_ids"]]
        pad_tok = self.tokenizer.pad_token_id
        max_length = len(tokens[0]) - min(
            map(lambda x: sum([int(t == pad_tok) for t in x]), tokens)
        )

        # truncate to `max_length`
        def maybe_truncate(x: Any, max_length: int):
            """truncate sequential attributes to `max_length`"""
            if not (isinstance(x, torch.Tensor) and len(x.shape) == 2):
                return x

            return x[:, :max_length]

        tensor_outpus = {
            k: maybe_truncate(v, max_length) for k, v in output.items()
        }
        return tensor_outpus

    @staticmethod
    @torch.no_grad()
    def compute_vectors_batch(
        key: str, model: Callable, batch: Batch
    ) -> Dict[str, Tensor]:
        """Compute one batch of vectors"""

        # move data to device
        if isinstance(model, torch.nn.Module):
            device = next(iter(model.parameters())).device
            batch = move_data_to_device(batch, device)

        # process with the model (Dense or Sparse)
        batch[key] = model(batch)

        # cast to numpy and return
        return {
            k: v.to(device="cpu").numpy() if isinstance(v, Tensor) else v
            for k, v in batch.items()
            if k != "_mode_"
        }

    def index(
        self,
        model: Optional[Callable] = None,
        index: Optional[str] = "faiss",
        filtering: Optional[str] = None,
        **kwargs,
    ):
        """
        Compute vectors (sparse or dense) for the whole dataset.

        :@param model: callable that returns a vector given the batch input.
        """
        if model is not None:
            self.dataset = self.dataset.map(
                partial(self.compute_vectors_batch, self.vectors_id, model),
                batched=True,
                batch_size=self.eval_batch_size,
                num_proc=1,
                desc="Computing corpus vectors",
            )
        if index == "faiss":
            self.dataset["train"].add_faiss_index(
                column=self.vectors_id, **kwargs
            )
        elif index == "bm25":
            es_create_index("corpus")

            # decide whether to filter text to only include medical relevant terms
            if filtering == "scispacy":
                raise NotImplementedError
            # self.dataset.set_format(type='numpy', columns=['document.attention_mask', 'document.idx', 'document.input_ids', 'document.passage_idx', 'document.passage_mask', 'document.vectors'])
            print(self.dataset["train"].column_names)
            print(self.dataset["train"]["document.text"][0])
            es_bulk("corpus", "book1", self.dataset["train"]["document.text"])

            # for name in self.dataset["train"].column_names:
            #     if torch.is_tensor(self.dataset["train"][name]):
            #         print(name)
            #         self.dataset["train"].map(lambda row: {name : row[name].numpy()})
            #         print(type(self.dataset["train"][name]))

            # subset = self.dataset.remove_columns(['document.attention_mask', 'document.idx',
            # 'document.input_ids','document.passage_idx','document.passage_mask','document.vectors'])
            # print(type(subset['train']))
            # print(subset['train'].column_names)
            # squad.add_elasticsearch_index(
            #     column="context",
            #     host="localhost",
            #     port="9200",
            #     es_index_name="corpus",
            # )
            print("trying to print index")
            # response = es.indices.exists(index="corpus")
            # print(response)

        else:
            raise NotImplementedError

    def query(
        self,
        query: Optional[str] = None,
        vector: Optional[Tensor] = None,
        k: int = 1,
        index: Optional[str] = "faiss",
        filtering: Optional[str] = None,
    ):
        """Query index given a input query"""
        if index == "faiss":
            # todo: this causes segmentation fault on MacOS, works fine on the cluster
            vector = vector.cpu().numpy()
            return self.dataset["train"].get_nearest_examples(
                self.vectors_id, vector, k=k
            )

        elif index == "bm25":
            return self.dataset["train"].get_nearest_examples(
                "document.text", query, k=k
            )

    def query_batch(self, vectors: Tensor, k: int = 1):
        """Query the faiss index given a batch of vector queries of shape (bs, h,)"""
        vectors = vectors.cpu().numpy()
        return self.dataset["train"].get_nearest_examples_batch(
            self.vectors_id, vectors, k=k
        )

    def val_dataloader(self):
        return self._eval_loader(
            Split.TRAIN
        )  # the dataset only have one split

    def test_dataloader(self):
        return self._eval_loader(
            Split.TRAIN
        )  # the dataset only have one split

    @staticmethod
    def take_subset(dataset: HgDataset) -> HgDataset:
        """Take a subset of the dataset and return."""
        if isinstance(dataset, DatasetDict):
            return DatasetDict(
                {
                    k: dset.select(range(n))
                    for n, (k, dset) in zip([10, 1, 1], dataset.items())
                }
            )
        elif isinstance(dataset, Dataset):
            return dataset.select(range(1))
        else:
            raise NotImplementedError

    def pprint(self):
        """Pretty print the dtaset"""
        rich.print(
            f">> Dataset: (use_subset={self.use_subset}): \n" f"{self.dataset}"
        )


class MedQaEnDataModule(CorpusDataModule):
    dset_script_path_or_id = meqa_en_corpus.__file__


class FzCorpusDataModule(CorpusDataModule):
    dset_script_path_or_id = fz_corpus.__file__
