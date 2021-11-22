import itertools
import json
import logging
import os
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

import rich
from datasets import Dataset
from datasets import DatasetDict
from datasets import load_dataset
from googleapiclient.http import MediaFileUpload
from rich.progress import track
from rich.status import Status

from fz_openqa.datamodules.builders import DatasetBuilder
from fz_openqa.datamodules.builders import MedQABuilder
from fz_openqa.datamodules.pipes.query_wiki_api import QueryWikiAPI
from fz_openqa.utils.gdrive_dl_manager import _create_service

logger = logging.getLogger(__name__)

CLIENT_SECRET_FILE = "./fz_openqa/utils/client-secrets.json"
API_NAME = "drive"
API_VERSION = "v3"
SCOPES = ["https://www.googleapis.com/auth/drive"]


class WikixMedQaCorpusBuilder(DatasetBuilder):
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
                title: idx for idx, title in enumerate(self.wikipedia_data["title"])
            }
        # Directory path to output Wikipedia corpus
        self.dataset_dict_path = dataset_dict_path

        # Set up GoogleDrive Service
        self.drive = _create_service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)

    def __call__(self, format: Optional[str] = None, **kwargs):
        with Status(f"Instantiating {self.dataset_builder.__module__}.."):
            dataset = self.dataset_builder(format=None, tokenizer=None)

        # process the whole dataset (extract wikipedia pages)
        dataset = self.extract_page_titles(dataset=dataset)
        rich.print(dataset)
        # build Wikipedia corpus to output
        json_list = self.build_wiki_corpus(dataset=dataset)
        rich.print(len(json_list))
        # exit()
        with open(os.path.join(self.dataset_dict_path, "wiki_corpus.jsonl"), mode="w") as fp:
            json.dump(json_list, fp)

        # new_dataset.save_to_disk(self.dataset_dict_path)
        file = self._upload_to_drive(
            path_to_file=os.path.join(self.dataset_dict_path, "wiki_corpus.jsonl")
        )

        rich.print(file)
        exit()

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
        self.title_index[page] = ""

    def extract_page_content(self, qst_idx: int, pages: List[str]) -> List[Dict]:
        """Extracts the page content of each Wikipedia page"""
        wiki_content = []
        for page_title in pages:
            self._update_title_index(page_title)
            wiki_idx = self.wikipedia_index.get(page_title)
            if wiki_idx:
                wiki_page = self.wikipedia_data.__getitem__(wiki_idx)
                wiki_content.append(
                    {"question.idx": qst_idx, "title": page_title, "text": wiki_page["text"]}
                )
        return wiki_content

    def build_wiki_corpus(self, dataset: DatasetDict) -> Dataset:
        """Builds the Wikipedia Corpus based on extracted Wikipedia pages
        Features: {"document.title", "document.text"}
        """
        # data_dict = {"document.title": [], "document.text": []}
        json_list = []
        for split, ds in dataset.items():
            for eg in track(ds, description=f"Iterating through the {split} dataset..."):
                eg["wiki.pages"] = list(
                    itertools.filterfalse(
                        lambda x: x in self.title_index.keys(), set(eg["wiki.pages"])
                    )
                )
                json_list.extend(
                    self.extract_page_content(qst_idx=eg["question.idx"], pages=eg["wiki.pages"])
                )
                # data_dict["document.title"].extend(titles)
                # data_dict["document.text"].extend(texts)

        return json_list

    def _upload_to_drive(self, path_to_file: str):

        file_id = "1h1SSvTWg7aW1g5-IVQy4_fd3D_yY74H2"
        content = MediaFileUpload(path_to_file, mimetype="*/*", resumable=True)

        file = self.drive.files().update(fileId=file_id, media_body=content).execute()

        return file
