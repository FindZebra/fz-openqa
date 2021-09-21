import random
import shutil
from functools import partial
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import datasets
import rich
import torch
from datasets import Split
from torch.functional import Tensor
from transformers import BatchEncoding
from transformers import PreTrainedTokenizerFast

from .base_dm import BaseDataModule
from .datasets import medqa
from .utils import add_spec_token
from .utils import HgDataset
from .utils import nested_list
from fz_openqa.datamodules.pipes.collate_fn import collate_and_pad_attributes
from fz_openqa.datamodules.pipes.collate_fn import collate_answer_options
from fz_openqa.datamodules.pipes.collate_fn import collate_nested_examples
from fz_openqa.datamodules.pipes.collate_fn import (
    collate_simple_attributes_by_key,
)
from fz_openqa.datamodules.pipes.collate_fn import (
    extract_and_collate_attributes_as_list,
)
from fz_openqa.tokenizers.static import ANS_TOKEN
from fz_openqa.tokenizers.static import QUERY_TOKEN
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.es_functions import es_search
from fz_openqa.utils.pretty import pretty_decode

# from fz_openqa.tokenizers.static import DOC_TOKEN

PT_SIMPLE_ATTRIBUTES = [
    "answer.target",
    "answer.n_options",
    "question.idx",
]


class MedQaDataModule(BaseDataModule):
    """A PyTorch Lightning DataModule for handling the MedQA questions and answers."""

    dset_script_path_or_id = (
        medqa.__file__  # HuggingFace dataset id or local path to script
    )
    split_ids = [
        datasets.Split.TRAIN,
        datasets.Split.VALIDATION,
        datasets.Split.TEST,
    ]  # split names
    text_fields = ["question", "answer", "synonyms"]
    pt_attributes = [
        "idx",
        "passage_idx",
        "input_ids",
        "attention_mask",
        "passage_mask",
    ]  # attributes to be converted into Tensors  # to be generated
    # attributes to be converted into Tensors

    def __init__(
        self,
        *,
        filter_synonyms: bool = True,
        top_n_synonyms: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filter_synonyms = filter_synonyms
        self.top_n_synonyms = (top_n_synonyms,)

    @staticmethod
    def tokenize_examples(
        examples: Dict[str, List[Any]],
        *,
        tokenizer: PreTrainedTokenizerFast,
        max_length: Optional[int],
        add_encoding_tokens: bool = True,
        **kwargs,
    ) -> Union[Dict, BatchEncoding]:
        """Tokenize a batch of examples and truncate if `max_length` is provided.
        examples = {
            attribute_name: list of attribute values
        }
        the output if of the form:
        output = {
            question.input_ids: [e.question.tokens for e in examples]
            question.attention_mask: [e.question.mask for e in examples]
            answer_0.input_ids: [e.answer_0.tokens for e in examples]
            answer_0.attention_mask: [e.answer_0.mask for e in examples]
            answer_1.input_ids: [e.answer_1.tokens for e in examples]
            answer_1.attention_mask: [e.answer_1.mask for e in examples]
            [...]
        }

        """
        tokenizer_kwargs = {
            "max_length": max_length,
            "return_token_type_ids": False,
            "add_special_tokens": True,
            "truncation": max_length is not None,
        }

        # process questions
        questions = (
            list(
                map(partial(add_spec_token, QUERY_TOKEN), examples["question"])
            )
            if add_encoding_tokens
            else examples["question"]
        )
        q_encodings = tokenizer(questions, **tokenizer_kwargs)

        # prepare the output
        output = {}

        # append the "question" prefix to the
        # "input_ids" and "attention masts from the q_encodings
        # and store them in output.
        for data, prefix in zip([q_encodings], ["question"]):
            for k, v in data.items():
                output[f"{prefix}.{k}"] = v

        # process answers
        add_answ_token = (
            partial(add_spec_token, ANS_TOKEN)
            if add_encoding_tokens
            else lambda x: x
        )
        output["answer.n_options"] = [len(ex) for ex in examples["answer"]]
        answer_encodings = tokenizer(
            [
                add_answ_token(choice)
                for choices in examples["answer"]
                for choice in choices
            ],
            **tokenizer_kwargs,
        )
        assert all(
            x == output["answer.n_options"][0]
            for x in output["answer.n_options"]
        )
        output.update(
            **{
                f"answer.{k}": nested_list(
                    v, stride=output["answer.n_options"][0]
                )
                for k, v in answer_encodings.items()
            }
        )

        return output

    def preprocess_dataset(self, dataset: HgDataset) -> HgDataset:
        """Apply processing steps to the dataset. Tokenization and formatting as PyTorch tensors"""
        # tokenize
        fn = partial(
            self.tokenize_examples,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            add_encoding_tokens=self.add_encoding_tokens,
        )
        dataset = dataset.map(
            fn, batched=True, num_proc=self.num_proc, desc="Tokenizing"
        )

        # to-do: add 'idx' column

        # rename text attributes
        for key in ["question", "answer"]:
            dataset = dataset.rename_column(key, f"{key}.text")

        # transform attributes to tensors
        attrs = ["input_ids", "attention_mask"]
        columns = ["question", "answer"]
        self.pt_attributes = [
            c
            for c in dataset.column_names["train"]
            if (any(a in c for a in attrs) and any(a in c for a in columns))
        ]
        self.pt_attributes += PT_SIMPLE_ATTRIBUTES
        dataset.set_format(
            type="torch", columns=self.pt_attributes, output_all_columns=True
        )

        return dataset

    def update_synonym_list(self, example):
        example["synonyms"] = example["synonyms"][: self.top_n_synonyms[0]]
        return example

    def filter_dataset(self, dataset: HgDataset) -> HgDataset:
        """Apply filtering operations to filter list of synonyms"""
        if self.filter_synonyms:
            dataset = dataset.map(self.update_synonym_list)

        return dataset

    @staticmethod
    def filter_question_id(ids: List[int], row: Dict[str, Any]) -> bool:
        return row["question.idx"] in ids

    @staticmethod
    def take_subset(dataset: HgDataset) -> HgDataset:
        """Take a subset of the dataset and return."""
        subset_size = {Split.TRAIN: 5, Split.VALIDATION: 2, Split.TEST: 2}
        for key, dset in dataset.items():
            questions_ids = dset["question.idx"]
            selected_ids = random.sample(questions_ids, k=subset_size[key])
            fn = partial(MedQaDataModule.filter_question_id, selected_ids)
            dataset[key] = dset.filter(fn)

        return dataset

    def collate_fn(self, examples: Any) -> Batch:
        return self.collate_fn_(self.tokenizer, examples)

    @staticmethod
    def collate_fn_(
        tokenizer: PreTrainedTokenizerFast,
        examples: Any,
    ) -> Batch:
        """The function that is used to merge examples into a batch.
        Concatenating sequences with different length requires padding them.

        The input data must be of the following structure:
        ```yaml
            - example_1:
                - sub_example_1:
                    - question.idx: a
                    - document.idx: x
                - sub_example_2:
                    - question.idx: a
                    - document.idx: y
            - example_2:
                - sub_example_1:
                    - question.idx: b
                    - document.idx: y
                - sub_example_2:
                    - question.idx: a
                    - document.idx: z
        ```
        If the input data is a List of examples, it will be converted bellow as a list of list of examples.

        Returns a dictionary with attributes:
        output = {
                question.idx: tensor of shape [N,],
                question.text: list of N texts
                question.input_ids: tensor of shape [N, T],
                question.attention_mask: tensor of shape [N, T],
                answer.text: nested list of [N, N_a] texts
                answer.input_ids: tensor of shape [N, N_a, T]
                answer.attention_mask: tensor of shape [N, N_a, T]
                answer.target: tensor of shape [N,]
                answer.n_options: tensor of hsape [N,]
        }
        """
        output = {}

        # convert as List of Lists if that's not the case (general case)
        if not isinstance(examples[0], list):
            examples = [[ex] for ex in examples]

        # collate the question and answers using the first example of each batch element
        first_examples = [ex[0] for ex in examples]
        output.update(
            **MedQaDataModule.collate_qa(first_examples, tokenizer=tokenizer)
        )

        # collate documents
        output.update(
            **collate_nested_examples(
                examples, tokenizer=tokenizer, key="document"
            )
        )

        return output

    @staticmethod
    def collate_qa(
        examples: List[Batch], *, tokenizer: PreTrainedTokenizerFast
    ) -> Batch:
        """collate the question and answer data"""

        # get the raw text questions, extract and collate
        examples, output = extract_and_collate_attributes_as_list(
            examples, attribute="text", key="question"
        )

        # collate simple attributes
        for k in ["idx", "answer.target", "answer.n_options"]:
            output[k] = collate_simple_attributes_by_key(
                examples, key=k, extract=True
            )

        # collate the questions attributes (question.input_ids, question.idx, ...)
        output.update(
            **collate_and_pad_attributes(
                examples, tokenizer=tokenizer, key="question", exclude=".text"
            )
        )

        # collate answer options
        output.update(**collate_answer_options(examples, tokenizer=tokenizer))

        return output

    def search_index(
        self,
        query: Optional[str] = None,
        vector: Optional[Tensor] = None,
        k: int = 1,
        index: Optional[str] = "faiss",
        filtering: Optional[str] = None,
        name: str = "corpus",
    ):
        """
        Query index given a input query

        :@param query:
        :@param vector:
        :@param k: integer that sets number of results to be queried.
        :@param index: string that determines which index to use (faiss or bm25).
        :@param filtering: string that determines whether SciSpacy filtering is used.
        :@param name: string for naming index 'group'.
        """
        if index == "faiss":
            # todo: this causes segmentation fault on MacOS, works fine on the cluster
            vector = vector.cpu().numpy()
            return self.dataset["train"].get_nearest_examples(
                self.vectors_id, vector, k=k
            )

        elif index == "bm25":
            return es_search(index_name=name, query=query, results=k)

        else:
            raise NotImplementedError

    def exact_method(self, batch: Batch) -> Batch:
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

        for i, query in enumerate(batch["question.text"]):
            response = self.search_index(query=query, k=100, index="bm25")

            positives = []
            negatives = []
            for hit in response["hits"]:
                if batch["answer.text"][i][0] in hit["_source"]["text"]:
                    positives.append(hit["_source"]["text"])
                else:
                    negatives.append(hit["_source"]["text"])

            if positives:
                out["data"].append(
                    {
                        "question": query,
                        "answer": batch["answer.text"][i][0],
                        "positive": positives[0],
                        "negatives": negatives[0:10],
                    }
                )

        return out

    def display_one_sample(self, example: Dict[str, torch.Tensor]):
        """Decode and print one example from the batch"""
        decode_kwargs = {
            "skip_special_tokens": False,
            "tokenizer": self.tokenizer,
        }
        console_width, _ = shutil.get_terminal_size()
        print("=== Sample ===")
        print(console_width * "-")
        print("* Question:")
        rich.print(
            pretty_decode(example["question.input_ids"], **decode_kwargs)
        )

        print(console_width * "-")
        rich.print("* Documents: ")
        for k in range(example["document.input_ids"].shape[0]):
            rich.print(
                f"-- document [magenta]{k}[/magenta] (is_positive={example['document.is_positive'][k]}, rank={example['document.rank'][k]}) --"
            )
            rich.print(
                pretty_decode(
                    example["document.input_ids"][k],
                    **decode_kwargs,
                )
            )
        print(console_width * "-")
        print("* Answer Choices:")
        idx = example["answer.target"]
        for i, an in enumerate(example["answer.input_ids"]):
            print(
                f"   - [{'x' if idx == i else ' '}] "
                f"{self.tokenizer.decode(an, **decode_kwargs).replace('[PAD]', '').strip()}"
            )
        print(console_width * "=")

    vectors_id = "document.vectors"
