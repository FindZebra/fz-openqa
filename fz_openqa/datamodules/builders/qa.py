from typing import Any
from typing import Dict
from typing import Optional

import dill  # type: ignore
from datasets import concatenate_datasets
from datasets import DatasetDict
from datasets import load_dataset
from loguru import logger
from warp_pipes import ApplyAsFlatten
from warp_pipes import CollateField
from warp_pipes import Expand
from warp_pipes import Gate
from warp_pipes import HasPrefix
from warp_pipes import HfDataset
from warp_pipes import Identity
from warp_pipes import In
from warp_pipes import Parallel
from warp_pipes import Reduce
from warp_pipes import RenameKeys
from warp_pipes import Sequential
from warp_pipes.support.datasets_utils import keep_only_columns

from .adapters import DATASET_ADAPTERS
from .hf_dataset import HfDatasetBuilder
from .preprocessing import DatasetPreprocessing
from .utils.format_row import format_row_concatenated_questions
from .utils.format_row import format_row_flat_questions
from fz_openqa.datamodules.generators import fz_queries
from fz_openqa.datamodules.generators import medqa
from fz_openqa.datamodules.generators import quality
from fz_openqa.datamodules.pipelines.preprocessing import FormatAndTokenize
from fz_openqa.datamodules.pipes.concat_fields import ConcatTokenFields
from fz_openqa.datamodules.pipes.query_expansion import QueryExpansion
from fz_openqa.datamodules.utils.transformations import set_index_column
from fz_openqa.tokenizers.static import ANS_TOKEN
from fz_openqa.tokenizers.static import DOC_TOKEN
from fz_openqa.tokenizers.static import QUERY_TOKEN

QA_DATASETS = {
    "medqa-us": (medqa.__file__, "us"),
    "medqa-tw": (medqa.__file__, "tw"),
    "quality": (quality.__file__, None),
    "race": ("race", "all"),
    "race-hard": ("race", "hard"),
    "race-middle": ("race", "middle"),
    "medmcqa": ("medmcqa", None),
    "fz-queries": (fz_queries.__file__, None),
}


class QaBuilder(HfDatasetBuilder):
    # HuggingFace dataset id or local path to script
    # these values are set dynamically in the __init__

    # nesting level of the question field
    nesting_level = 0

    # name of the attributes that will be converted to
    # tensors in the preprocessing function
    pt_attributes = [
        "question.input_ids",
        "question.attention_mask",
        "question.idx",
        "question.row_idx",
        "answer.input_ids",
        "answer.attention_mask",
        "answer.target",
        "document.proposal_score",
    ]

    # number of options
    n_options = 4

    # output columns
    column_names = [
        "answer.text",
        "answer.input_ids",
        "answer.attention_mask",
        "answer.target",
        "question.text",
        "question.metamap",
        "question.input_ids",
        "question.attention_mask",
    ]

    def __init__(
        self,
        *args,
        n_query_tokens: int = 1,
        n_answer_tokens: int = 1,
        query_expansion: Optional[int] = None,
        dset_name: str = "medqa-us",
        drop_documents: bool = True,
        preprocessing_op: Optional[DatasetPreprocessing] = None,
        **kwargs,
    ):
        super(QaBuilder, self).__init__(*args, **kwargs)
        self.query_expansion = query_expansion
        self.n_query_tokens = n_query_tokens
        self.n_answer_tokens = n_answer_tokens
        self.drop_documents = drop_documents
        self.preprocessing_op = preprocessing_op

        # set the dataset attributes
        for dn in dset_name.split("+"):
            if dn not in QA_DATASETS:
                raise ValueError(f"Unknown dataset {dn}, available: {list(QA_DATASETS.keys())}")

        self.dset_name = dset_name

    def load_one_dataset(self, dset_name, **kwargs):
        # load the dataset
        dset_args = QA_DATASETS.get(dset_name, (dset_name,))
        dataset = load_dataset(*dset_args, **kwargs)

        # adapt the dataset
        if dset_name in DATASET_ADAPTERS:
            adapter = DATASET_ADAPTERS[dset_name]()
            dataset, corpus = adapter(dataset, num_proc=self.num_proc)
        return dataset

    def load_base_dataset(self) -> DatasetDict:
        """
        Loads the base dataset. Multiple dataset names can be passed
        using "+" as a separator. e.g. "tw+us"
        """
        dset_names = sorted(self.dset_name.split("+"))
        for dset_name in dset_names:
            script_id, name = QA_DATASETS[dset_name]
            logger.info(f"Loading dataset `{script_id}` with `{name}`")

        # load the base datasets
        kwargs = {"cache_dir": self.cache_dir}
        dsets = {dset_name: self.load_one_dataset(dset_name, **kwargs) for dset_name in dset_names}

        # apply preprocessing operators
        if self.preprocessing_op is not None:
            dsets = {
                dset_name: self.preprocessing_op(
                    d,
                    num_proc=self.num_proc,
                    dset_name=dset_name,
                )
                for dset_name, d in dsets.items()
            }

        # convert the dict of datasets to a list of datasets
        dsets = list(dsets.values())

        # concatenate the datasets
        if len(dsets) == 1:
            return dsets[0]
        else:
            # check that all datasets have the same splits
            splits = set(dsets[0].keys())
            if not all(set(dset.keys()) == splits for dset in dsets):
                raise ValueError(
                    f"Datasets do not all have the same splits. "
                    f"Found {[set(dset.keys()) for dset in dsets]}"
                )

            # only keep the columns that appear in all datasets
            shared_columns = set.intersection(
                *(set(ds.column_names) for dset in dsets for ds in dset.values())
            )
            dsets = [keep_only_columns(dset, list(shared_columns)) for dset in dsets]

            # concatenate and add the `dataset.idx` column
            dsets_dict = DatasetDict()
            for split in splits:
                split_dsets = [d[split] for d in dsets]
                split_dsets = [
                    d.add_column("dataset.idx", [i] * len(d)) for i, d in enumerate(split_dsets)
                ]
                split_dsets = concatenate_datasets(split_dsets)
                dsets_dict[split] = split_dsets

            return dsets_dict

    def preprocess_dataset(self, dataset: HfDataset) -> HfDataset:
        """Apply processing steps to the dataset.
        Tokenization and formatting as PyTorch tensors"""

        if self.drop_documents:
            cols = list({c for c in dataset.column_names if "document." in c})
            logger.info(f"Dropping document columns {cols}")
            dataset = dataset.remove_columns(cols)

        # Tokenize the text fields (questions, answers, and documents, if any)
        if self.tokenizer:
            one_split = next(iter(dataset.keys()))
            has_document_columns = any("document." in c for c in dataset[one_split].column_names)
            has_answer_columns = any("answer." in c for c in dataset[one_split].column_names)
            dataset = dataset.map(
                Parallel(
                    self.get_question_tokenizer_pipe(),
                    Gate(has_answer_columns, self.get_answer_tokenizer_pipe()),
                    Gate(has_document_columns, self.get_document_tokenizer_pipe()),
                ),
                batched=True,
                num_proc=self.num_proc,
                desc="Tokenizing",
            )

        # add an index column
        dataset = set_index_column(dataset, key="question.row_idx")

        return dataset

    def get_answer_tokenizer_pipe(self):
        return FormatAndTokenize(
            prefix="answer.",
            text_formatter=self.text_formatter,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            add_qad_tokens=self.add_qad_tokens,
            qad_tokens=self.n_answer_tokens * [ANS_TOKEN],
            shape=[-1, self.n_options],
        )

    def get_document_tokenizer_pipe(self):
        return FormatAndTokenize(
            prefix="document.",
            text_formatter=self.text_formatter,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            add_qad_tokens=self.add_qad_tokens,
            qad_tokens=[DOC_TOKEN],
            shape=None,
        )

    def get_question_tokenizer_pipe(self):
        """create a Pipe to tokenize the questions."""

        if self.query_expansion is not None:
            query_expansion_pipe = QueryExpansion(
                question_length=self.query_expansion, tokenizer=self.tokenizer, update=True
            )
        else:
            query_expansion_pipe = None

        return Sequential(
            FormatAndTokenize(
                prefix="question.",
                text_formatter=self.text_formatter,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                add_qad_tokens=self.add_qad_tokens,
                add_special_tokens=self.add_special_tokens,
                qad_tokens=self.n_query_tokens * [QUERY_TOKEN],
                shape=None,
            ),
            query_expansion_pipe,
        )

    def _get_collate_pipe(self, nesting_level=None):
        """get the raw text questions, extract and collate"""
        return Parallel(
            CollateField(
                "question", tokenizer=self.tokenizer, nesting_level=0, id="collate-questions"
            ),
            CollateField(
                "document",
                tokenizer=self.tokenizer,
                nesting_level=0,
                id="collate-documents",
                include_only=["text", "input_ids", "attention_mask"],
            ),
            CollateField(
                "answer",
                nesting_level=0,
                exclude=["input_ids", "attention_mask"],
                to_tensor=["target"],
                id="collate-answer-attributes",
            ),
            CollateField(
                "answer",
                tokenizer=self.tokenizer,
                nesting_level=1,
                include_only=["input_ids", "attention_mask"],
                id="pad-answer-tokens",
            ),
            CollateField(
                "answer",
                tokenizer=self.tokenizer,
                nesting_level=1,
                include_only=["input_ids", "attention_mask"],
                id="pad-answer-tokens",
            ),
        )

    def format_row(self, row: Dict[str, Any], **kwargs) -> str:
        return format_row_flat_questions(row, tokenizer=self.tokenizer, **kwargs)


class ConcatQaBuilder(QaBuilder):
    """A MedQa dataset with concatenated questions and answers"""

    # nesting level of the question field
    nesting_level = 1

    # name of the attributes that will be converted to
    # tensors in the preprocessing function
    pt_attributes = [
        "question.input_ids",
        "question.attention_mask",
        "question.document_idx",
        "answer.target",
        "document.proposal_score",
    ]

    # output columns
    column_names = [
        "question.text",
        "question.input_ids",
        "question.attention_mask",
        "answer.text",
        "answer.target",
        "document.input_ids",
        "document.attention_mask",
    ]

    def preprocess_dataset(self, dataset: HfDataset) -> HfDataset:
        """Apply processing steps to the dataset.
        Tokenization and formatting as PyTorch tensors"""

        # concat question and answers
        dataset = dataset.map(
            self.get_tokenize_concat_qa_pipe(),
            batched=True,
            num_proc=self.num_proc,
            desc="Tokenizing and concatenating questions and answers",
        )

        # add an index column
        dataset = set_index_column(dataset, key="question.row_idx")

        return dataset

    def get_tokenize_concat_qa_pipe(self):
        # register features that also need to be expanded to match the concatenated shape
        text_keys = ["question.text", "answer.text"]
        question_features = ["question.text", "question.input_ids", "question.attention_mask"]
        additional_question_features = ["question.document_idx"]

        # register the tokens that prefix the question
        q_start_tokens = []
        if self.add_qad_tokens:
            q_start_tokens.extend(self.n_query_tokens * [QUERY_TOKEN])

        # register the tokens used to separate the questions and answers, this is used
        # for the GPT2 tokenizer which doesn't support CLS tokens
        if self.tokenizer.cls_token is not None:
            # no need to add them, the CLS token is already provided
            sep_tokens = None
        else:
            space_token_id = self.tokenizer.encode(" ", add_special_tokens=False)[0]
            sep_tokens = {
                "input_ids": [space_token_id],
                "attention_mask": [1],
                "token_type_ids": [0],
                "offset_mapping": [[-1, -1]],
            }

        # return the final pipe
        return Sequential(
            self.text_formatter.copy(text_key=text_keys, update=True),
            # tokenize the questions and the answers
            Parallel(
                self.get_question_tokenizer_pipe(),
                self.get_answer_tokenizer_pipe(),
                Identity(input_filter=In([*additional_question_features, *text_keys])),
            ),
            Expand(
                axis=1,
                n=self.n_options,
                update=True,
                input_filter=In([*question_features, *additional_question_features]),
            ),
            # concat the answer text with the question
            ApplyAsFlatten(
                ConcatTokenFields(
                    fields=["question", "answer"],
                    master_key="input_ids",
                    new_field="question",
                    drop_start_tokens=[self.tokenizer.cls_token_id],
                    keep_end_tokens=[self.tokenizer.sep_token_id],
                    max_length=self.max_length,
                    sep_tokens=sep_tokens,
                ),
                level=1,
                input_filter=Reduce(
                    *[HasPrefix(field) for field in ["question", "answer"]], reduce_op=any
                ),
                update=True,
            ),
            RenameKeys({"answer.text": "question.answer_text"}),
            input_filter=In(["question.text", "answer.text", *additional_question_features]),
        )

    def _get_collate_pipe(self, nesting_level=1):
        return Parallel(
            CollateField(
                "question",
                exclude=["input_ids", "attention_mask", "token_type_ids"],
                to_tensor=["document_idx"],
                nesting_level=0,
                id="collate-questions",
            ),
            CollateField(
                "answer",
                nesting_level=0,
                exclude=["input_ids", "attention_mask"],
                to_tensor=["target"],
                id="collate-answer-attributes",
            ),
            CollateField(
                "question",
                tokenizer=self.tokenizer,
                nesting_level=nesting_level,
                include_only=["input_ids", "offset_mapping", "attention_mask", "token_type_ids"],
                id="pad-nested-question-tokens",
            ),
        )

    def format_row(self, row: Dict[str, Any], **kwargs) -> str:
        return format_row_concatenated_questions(row, tokenizer=self.tokenizer, **kwargs)
