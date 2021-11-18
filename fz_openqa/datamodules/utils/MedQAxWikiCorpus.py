import itertools
import logging
import os
import pickle
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

import rich
from datasets import DatasetDict
from datasets import load_dataset
from pyarrow._dataset import Dataset
from rich.progress import track
from rich.status import Status

from fz_openqa.datamodules.builders import DatasetBuilder
from fz_openqa.datamodules.builders import MedQABuilder
from fz_openqa.datamodules.pipes import BlockSequential
from fz_openqa.datamodules.pipes import FilterKeys
from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.datamodules.pipes import PrintBatch
from fz_openqa.datamodules.pipes import Sequential
from fz_openqa.datamodules.pipes import UpdateWith
from fz_openqa.datamodules.pipes.extract_wiki_page import ExtractWikiPage
from fz_openqa.datamodules.pipes.query_wiki_api import QueryWikiAPI
from fz_openqa.datamodules.utils.map_with_fingerprint import MapWithFingerprint
from fz_openqa.datamodules.utils.typing import HfDataset

logger = logging.getLogger(__name__)


class WikixMedQaCorpusBuilder(DatasetBuilder):
    column_names = MedQABuilder.column_names + [
        "idx",
        "title",
        "content",
    ]

    def __init__(
        self,
        *,
        dataset_builder: MedQABuilder,
        query_articles: Callable = QueryWikiAPI(text_key="answer.text"),
        num_proc: int = 4,
        batch_size: int = 10,
        dataset_dict_path: str = os.getcwd(),
        **kwargs,
    ):
        super(WikixMedQaCorpusBuilder, self).__init__(cache_dir=None, **kwargs)

        self.dataset_builder = dataset_builder
        self.query_articles = query_articles
        self.title_index = {}

        self.num_proc = num_proc
        self.batch_size = batch_size

        with Status("Downloding Wikipedia dump..."):
            self.wikipedia_data = load_dataset("wikipedia", "20200501.en", split="train")

        self.dataset_dict_path = dataset_dict_path

    def __call__(self, format: Optional[str] = None, **kwargs):
        with Status(f"Instantiating {self.dataset_builder.__module__}.."):
            dataset = self.dataset_builder(format=None, tokenizer=None)

        dataset = self.extract_page_titles(dataset=dataset)

        dataset = self.extract_page_content(dataset=dataset)

        new_dataset = self.build_wiki_corpus(dataset=dataset)

        new_dataset.save_to_disk(self.dataset_dict_path)
        rich.print(f"[green]Wikipedia Corpus was successfully saved to {self.dataset_dict_path}")

    def extract_page_titles(self, dataset: DatasetDict) -> DatasetDict:
        dataset = dataset.map(
            self.query_articles,
            num_proc=self.num_proc,
            batched=True,
            batch_size=self.batch_size,
            desc="Search for Wikipedia pages",
        )

        return dataset

    def extract_page_content(self, dataset: DatasetDict) -> DatasetDict:
        dataset.map(
            ExtractWikiPage(wiki_data=self.wikipedia_data, query_key="wiki.pages"),
            num_proc=self.num_proc,
            batched=True,
            batch_size=self.batch_size,
            desc="Extract content of Wikipedia pages",
        )

    @staticmethod
    def build_wiki_corpus(dataset: DatasetDict) -> Dataset:
        json_list = []
        for split, ds in dataset.items():
            for batch in track(ds, description=f"Iterating through the {split} dataset..."):
                # pages = list(itertools.filterfalse(
                #    lambda x: x in self.title_index.keys(), pages)
                # )
                json_list.extend(batch["wiki.pages"])
            rich.print(f"[red]{len(json_list)}")

        rich.print(f"[red]{len(json_list)}")
        return Dataset(el for el in json_list)
