import string
from functools import partial
from typing import Optional

from datasets import DatasetDict

from fz_openqa.datamodules.builders.adapters.base import DatasetAdapter
from fz_openqa.datamodules.builders.adapters.extract_corpus import extract_corpus
from fz_openqa.datamodules.builders.adapters.utils import set_document_row_idx
from fz_openqa.datamodules.pipes import Apply
from fz_openqa.datamodules.utils.transformations import set_constant_column
from fz_openqa.datamodules.utils.transformations import set_index_column
from fz_openqa.datamodules.utils.typing import HfDataset

RACE_COL_MAPPING = {
    "example_id": "document.uid",
    "article": "document.text",
    "answer": "answer.target",
    "question": "question.text",
    "options": "answer.text",
}


class RaceAdapter(DatasetAdapter):
    """Adapt the RACE dataset to the `fz_openqa` format."""

    def __call__(self, dataset: DatasetDict, **kwargs) -> (DatasetDict, Optional[HfDataset]):
        # rename column
        dataset = DatasetDict({s: d.rename_columns(RACE_COL_MAPPING) for s, d in dataset.items()})

        # set the labels registered as [A, B, C, D] to 0, 1, 2, 3
        dataset = dataset.map(
            Apply({"answer.target": string.ascii_uppercase.index}, element_wise=True),
            batched=True,
            **kwargs,
            desc="Converting A,B,C,D labels to indices",
        )

        # extract the Corpus
        corpus, document_keys = extract_corpus(dataset, key="document.uid")
        corpus = set_index_column(corpus, key="document.idx")
        # set an empty title
        corpus = set_constant_column(corpus, key="document.title", value="")

        # add the `question.document_idx`
        dataset = dataset.map(
            partial(set_document_row_idx, doc_uid_key="document.uid", document_keys=document_keys),
            batched=False,
            desc="Add document idx row",
            **kwargs,
        )

        # filter documents columns
        dataset = DatasetDict(
            {
                s: d.remove_columns([c for c in d.column_names if "document." in c])
                for s, d in dataset.items()
            }
        )
        cols = [c for c in corpus.column_names if "document." not in c] + ["document.uid"]
        corpus = corpus.remove_columns(cols)

        return dataset, corpus
