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
from datasets import Dataset
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
from fz_openqa.utils.functional import infer_batch_size

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
        # Pipe to query potential Wikipedia pages (e.g. Wikipedia API, SpikeX)
        self.query_articles = query_articles
        # Index applied to catch already queried Wikipedia pages
        self.title_index = {}

        self.num_proc = num_proc
        self.batch_size = batch_size

        with Status("Downloading Wikipedia dump..."):
            self.wikipedia_data = load_dataset("wikipedia", "20200501.en", split="train")
            # Index to look up Wikipedia pages and extract page content
            self.wikipedia_index = {
                title:  idx for idx, title in enumerate(self.wikipedia_data['title'])
            }
        # Directory path to output Wikipedia corpus
        self.dataset_dict_path = dataset_dict_path

    def __call__(self, format: Optional[str] = None, **kwargs):
        with Status(f"Instantiating {self.dataset_builder.__module__}.."):
            dataset = self.dataset_builder(format=None, tokenizer=None)

        # process the whole dataset (extract wikipedia pages)
        dataset = self.extract_page_titles(dataset=dataset)
        # build Wikipedia corpus to output
        new_dataset = self.build_wiki_corpus(dataset=dataset)

        exit()
        new_dataset.save_to_disk(self.dataset_dict_path)
        rich.print(f"[green]Wikipedia Corpus was successfully saved to {self.dataset_dict_path}")

    def extract_page_titles(self, dataset: DatasetDict) -> DatasetDict:
        """Extracts a list of Wikipedia pages for each question"""
        dataset = dataset.map(
            self.query_articles,
            num_proc=self.num_proc,
            batched=True,
            batch_size=self.batch_size,
            desc="Search for Wikipedia pages",
        )

        return dataset

    def _update_title_index(self, page: str):
        """Updates title index to catch already queried Wikipedia pages"""
        self.title_index[page] = ''

    def extract_page_content(self, pages: List[str]) -> DatasetDict:
        """Extracts the page content of each Wikipedia page"""
        titles = []
        texts = []
        for page_title in pages:
            self._update_title_index(page_title)
            wiki_idx = self.wikipedia_index.get(page_title)
            if wiki_idx:
                wiki_page = self.wikipedia_data.__getitem__(wiki_idx)
                titles.append(page_title)
                texts.append(wiki_page["text"])
        return titles, texts

    def build_wiki_corpus(self, dataset: DatasetDict) -> Dataset:
        """Builds the Wikipedia Corpus based on extracted Wikipedia pages
                Features: {"document.title", "document.text"}
        """
        data_dict = {"document.title": [], "document.text": []}
        for split, ds in dataset.items():
            for eg in track(ds, description=f"Iterating through the {split} dataset..."):
                eg['wiki.pages'] = list(itertools.filterfalse(
                     lambda x: x in self.title_index.keys(), set(eg['wiki.pages']))
                 )
                titles, texts = self.extract_page_content(pages=eg['wiki.pages'])
                data_dict["document.title"].extend(titles)
                data_dict["document.text"].extend(texts)

        rich.print(f"[red]{len(data_dict['document.title'])}")
        return Dataset.from_dict(data_dict)
