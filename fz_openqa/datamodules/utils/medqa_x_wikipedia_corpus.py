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
from apiclient import errors
from rich.progress import track
from rich.status import Status

from fz_openqa.datamodules.builders import DatasetBuilder
from fz_openqa.datamodules.builders import MedQABuilder
from fz_openqa.datamodules.pipes.query_wiki_api import QueryWikiAPI
from fz_openqa.utils.gdrive_dl_manager import _create_service

log = logging.getLogger(__name__)

CLIENT_SECRET_FILE = f"{os.getcwd()}/fz_openqa/utils/client-secrets.json"
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
        file_name: str = "wikipedia_corpus_v2.jsonl",
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
        self.file_name = file_name

        # Set up GoogleDrive Service
        self.drive = _create_service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)

    def __call__(self, format: Optional[str] = None, **kwargs):
        with Status(f"Instantiating {self.dataset_builder.__module__}.."):
            dataset = self.dataset_builder(format=None, tokenizer=None)

        # process the whole dataset (extract wikipedia pages)
        dataset = self.extract_page_titles(dataset=dataset)
        # build Wikipedia corpus to output
        json_list = self.build_wiki_corpus(dataset=dataset)

        with Status(f"Uploading file to Google Drive.."):
            with open(os.path.join(self.dataset_dict_path, self.file_name), mode="w") as fp:
                json.dump(json_list, fp)

            file = self._upload_to_drive(
                path_to_file=os.path.join(self.dataset_dict_path, self.file_name)
            )

        log.info(f"Uploaded file to Google Drive: <{file}>")
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

        return json_list

    def _retrieve_all_files(self, folder_id: str = "1mxQF7zm85cgP8jIvuRokCxopEDwmFlHb") -> Dict:
        """Retrieve a Dict of File resources.

        Args:
        service: Drive API service instance.
        Returns:
        Dict of File resources.
        """
        result = {}
        try:
            param = {
                'q': f"'{folder_id}' in parents and trashed=false"
            }
            files = self.drive.files().list(**param).execute()

            for f in files['files']:
                result[f['name']] = f['id']
        except (errors.HttpError, errors) as e:
            print('An error occurred: %s' % e)
        return result

    def _upload_to_drive(self, path_to_file: str):
        """ Update or create file on gdrive based on whether file name is in file list """
        file_name = path_to_file.split('/')[-1]
        #
        file_list = self._retrieve_all_files(folder_id="1mxQF7zm85cgP8jIvuRokCxopEDwmFlHb")

        content = MediaFileUpload(path_to_file, mimetype="*/*", resumable=True)
        if file_name in file_list.keys():
            file_id = file_list[file_name]
            # Update existing file based on id
            file = self.drive.files().update(fileId=file_id, media_body=content).execute()

        else:
            file_metadata = {
                'name': file_name,
                'parents': ['1mxQF7zm85cgP8jIvuRokCxopEDwmFlHb'],
                'mimetype': '*/*',
            }
            # Create new file
            file = self.drive.files().create(body=file_metadata, media_body=content).execute()

        return file
