from enum import Enum
from typing import Optional

from datasets import Dataset
from datasets import DatasetDict
from pydantic import BaseModel
from warp_pipes import Index


class OpenQaDataset(DatasetDict):
    def __init__(
        self,
        *,
        dataset: DatasetDict,
        corpus: Optional[Dataset],
        index: Optional[Index],
    ):
        super(OpenQaDataset, self).__init__(dataset)
        self.corpus = corpus
        self.index = index

    def new(self, *, dataset: DatasetDict) -> "OpenQaDataset":
        return OpenQaDataset(dataset=dataset, corpus=self.corpus, index=self.index)

    def __repr__(self):
        u = f"{self.__class__.__name__}:\n"
        u += f" - dataset={super().__repr__()}\n"
        u += f" - corpus={self.corpus}\n"
        u += f" - index={self.index}\n"
        return u


class OpenQaConfig(BaseModel):
    question_nesting_level: int
    document_nesting_level: int


class Scenario(Enum):
    none = "none"
    language_modelling = "language-modelling"
    generative_qa = "generative-qa"
    multiple_choice_qa = "multiple-choice-qa"
    multiple_choice_concat_qa = "multiple-choice-concat-qa"
    multiple_choice_flat_concat_qa = "multiple-choice-flat_concat-qa"
