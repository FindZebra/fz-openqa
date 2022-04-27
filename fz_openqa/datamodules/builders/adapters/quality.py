from functools import partial
from typing import Optional

from datasets import DatasetDict
from datasets import Split
from loguru import logger

from fz_openqa.datamodules.builders.adapters.base import DatasetAdapter
from fz_openqa.datamodules.builders.adapters.extract_corpus import extract_corpus
from fz_openqa.datamodules.builders.adapters.utils import set_document_row_idx
from fz_openqa.datamodules.utils.transformations import set_constant_column
from fz_openqa.datamodules.utils.transformations import set_index_column
from fz_openqa.datamodules.utils.typing import HfDataset

QUALITY_COL_MAPPING = {}


class QualityAdapter(DatasetAdapter):
    """Adapt the QuALITY dataset to the `fz_openqa` format."""

    def __call__(self, dataset: DatasetDict, **kwargs) -> (DatasetDict, Optional[HfDataset]):
        # extract the Corpus
        corpus, document_keys = extract_corpus(dataset, key="document.uid")
        corpus = set_index_column(corpus, key="document.idx")

        # set an empty title
        if "document.title" not in dataset:
            corpus = corpus.remove_columns(["document.title"])
        corpus = set_constant_column(corpus, key="document.title", value="")

        # add the `question.document_idx`
        dataset = dataset.map(
            partial(
                set_document_row_idx,
                doc_uid_key="document.uid",
                document_keys=document_keys,
                output_key="question.document_idx",
            ),
            batched=False,
            desc="Add document idx row",
            **kwargs,
        )

        # filter columns (document / question)
        dataset = DatasetDict(
            {
                s: d.remove_columns([c for c in d.column_names if "document." in c])
                for s, d in dataset.items()
            }
        )
        cols = [c for c in corpus.column_names if "document." not in c] + ["document.uid"]
        corpus = corpus.remove_columns(cols)

        # replace the test split
        if len(dataset[Split.TEST]) == 0 or Split.TEST not in dataset.keys():
            logger.warning(
                f"> {Split.TEST} split not found, replacing with {Split.VALIDATION} split"
            )
            dataset[Split.TEST] = dataset[Split.VALIDATION]

        return dataset, corpus
