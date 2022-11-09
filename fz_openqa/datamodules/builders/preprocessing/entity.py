from typing import Optional

import datasets
from datasets import Dataset
from loguru import logger

from fz_openqa.datamodules.builders.preprocessing import DatasetPreprocessing
from fz_openqa.datamodules.pipes import SciSpaCyFilter


class EntityPreprocessing(DatasetPreprocessing):
    def __init__(
        self,
        *,
        text_key: str = "question.text",
        spacy_model: str = "en_core_sci_lg",
        only_train_set: bool = True,
        update_dataset: bool = False,
        **kwargs,
    ):
        super().__init__(only_train_set=only_train_set, **kwargs)
        logger.info(f"Initializing {type(self).__name__} " f"with spacy_model: {spacy_model}")
        self.spacy_model = spacy_model
        self.update_dataset = update_dataset
        self.op = SciSpaCyFilter(text_key=text_key, model_name=spacy_model)

    def preprocess(self, dataset: Dataset, **kwargs) -> Dataset:
        # extract entities
        new_dataset = dataset.map(
            self.op,
            desc=f"Extracting entities from the questions using " f"{self.spacy_model}",
            batched=True,
            **kwargs,
        )

        # concatenate the datasets
        if self.update_dataset:
            dataset = datasets.concatenate_datasets([dataset, new_dataset])

        return dataset
