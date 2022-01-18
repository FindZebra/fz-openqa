import json
import logging
import os
from copy import copy
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Union

import rich
from datasets import concatenate_datasets
from datasets import Dataset
from datasets import DatasetDict
from datasets import load_dataset
from datasets import Split
from rich.progress import track

from fz_openqa.datamodules.builders import DatasetBuilder
from fz_openqa.datamodules.builders import QaBuilder
from fz_openqa.datamodules.builders.utils.gdrive import Gdrive
from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.datamodules.pipes.query_wiki_api import QueryWikiAPI
from fz_openqa.datamodules.utils.dataset import get_column_names
from fz_openqa.utils.datastruct import Batch

log = logging.getLogger(__name__)

CLIENT_SECRET_FILE = f"{os.getcwd()}/fz_openqa/utils/client-secrets.json"
API_NAME = "drive"
API_VERSION = "v3"
SCOPES = ["https://www.googleapis.com/auth/drive"]


class ExtractPageContent(Pipe):
    """Extract the wikipedia page content given its title."""

    def __init__(
        self,
        wikipedia_dataset: Dataset,
        wikipedia_index: Dict[str, int],
        title_key: str = "wiki.page",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.wikipedia_dataset = wikipedia_dataset
        self.wikipedia_index = wikipedia_index
        self.title_key = title_key

    def __call__(self, batch: Batch, **kwargs) -> Batch:
        titles = batch[self.title_key]

        # get the index of the wiki article within the wikipedia article, None otherwise
        indices = [self.wikipedia_index.get(title, None) for title in titles]
        original_indexes, non_none_indices = zip(
            *[(k, i) for k, i in enumerate(indices) if i is not None]
        )

        # query the index for the wikipedia article
        wiki_rows = self.wikipedia_dataset.select(non_none_indices, keep_in_memory=True)

        # add the retrieved article to the output, when available
        wiki_texts = wiki_rows["text"]
        output = [None] * len(indices)
        for i, txt in zip(original_indexes, wiki_texts):
            output[i] = txt

        return {"wiki.text": output}


class FlattenWikiPages(Pipe):
    """
    Flatten the data structure such that there is one wikipedia article per row.
    E.g. from structure A: {'question.idx: [1,2, 'wiki.pages: [[a,b,c], [x,y,z]]}
    to the flatten structured B: {'question.idx: [1,1,1,2,2,2], 'wiki.pages':[a,b,c,x,y,z]}"""

    def __init__(
        self, wiki_page_key: str = "wiki.pages", index_key: str = "question.idx", **kwargs
    ):
        super().__init__(**kwargs)
        self.wiki_page_key = wiki_page_key
        self.index_key = index_key

    def __call__(self, batch: Batch, **kwargs) -> Batch:
        pages = batch[self.wiki_page_key]
        indices = batch[self.index_key]

        new_pages = []
        new_indices = []
        for i in range(len(indices)):
            pages_i = pages[i]
            index = indices[i]
            for page in pages_i:
                new_pages.append(page)
                new_indices.append(index)

        return {self.wiki_page_key: new_pages, self.index_key: new_indices}


class AddSplitInfo:
    """Add the split column"""

    def __init__(self, split: Split, **kwargs):
        super().__init__(**kwargs)
        self.split = str(split)

    def __call__(self, batch: Batch, **kwargs) -> Batch:
        x = list(batch.values())[0]
        return {"split": [self.split for _ in range(len(x))]}


class WikixMedQaCorpusBuilder(DatasetBuilder):
    """Builds a corpus of related Wikipedia articles
    w.r.t. dataset containing pairs of (question, answer).

    Parameters
    __________
    dataset_builder
        Dataset containing pairs of (questiosn, answer)
    query_articles
        Pipe to extract Wikipedia page titles based on answer string
    num_proc
        Max number of processes when generating cache
    batch_size
        Number of examples per batch provided to function
    cache_dir
        Provide the name of a path for the cache file
    file_name
        Provide the name of file uploaded to Google Drive
    upload_to_drive
        Allow dataset to get uploaded to Google Drive
    """

    def __init__(
        self,
        *,
        dataset_builder: QaBuilder,
        query_articles: Callable = QueryWikiAPI(text_key="answer.text"),
        num_proc: int = 4,
        batch_size: int = 10,
        cache_dir: str = Path(os.getcwd()) / "cache",
        file_name: Optional[Union[str, Path]] = None,
        upload_to_drive: bool = False,
        **kwargs,
    ):
        super(WikixMedQaCorpusBuilder, self).__init__(cache_dir=None, **kwargs)

        self.dataset_builder = dataset_builder
        # Pipe to query potential Wikipedia pages (e.g. Wikipedia API, SpikeX)
        self.query_articles = query_articles
        # Index applied to catch already queried Wikipedia pages
        self.title_index = {}

        self.map_kwargs = {"batched": True, "batch_size": batch_size, "num_proc": num_proc}

        log.info("Downloading Wikipedia dump...")
        self.wikipedia_data = load_dataset("wikipedia", "20200501.en", split="train")
        # Index to look up Wikipedia pages and extract page content
        self.wikipedia_index = {
            title: idx for idx, title in enumerate(self.wikipedia_data["title"])
        }

        # Directory path to output Wikipedia corpus
        self.cache_dir = Path(cache_dir)
        if file_name is not None:
            self.output_file = self.cache_dir / file_name

        # Set up GoogleDrive Service
        if upload_to_drive:
            self.drive = Gdrive()
        else:
            self.drive = None

    def __call__(self, format: Optional[str] = None, **kwargs):
        log.info(f"Instantiating {self.dataset_builder.__module__}..")
        dataset = self.dataset_builder(format=None, tokenizer=None)

        # process the whole dataset (extract wikipedia pages)
        dataset = self.extract_page_titles(dataset=dataset)

        # build Wikipedia corpus to output
        dataset = self.build_wiki_corpus(dataset=dataset)

        # write to disk
        if self.output_file is not None:
            log.info(f"Save the dataset to file n_lines={len(dataset)}")
            dataset.save_to_disk(str(self.output_file))

        # upload to drive
        if self.drive is not None:
            log.info("Uploading file to Google Drive..")
            file = self.drive.upload_to_drive(path_to_file=str(self.output_file))
            log.info(f"Uploaded file to Google Drive: <{file}>")

        return dataset

    def extract_page_titles(self, dataset: DatasetDict) -> DatasetDict:
        """Extracts a list of Wikipedia pages for each question.
        Only keep "wiki.pages" and "question.idx" """
        dataset = dataset.map(
            self.query_articles,
            **self.map_kwargs,
            desc="Search for Wikipedia pages",
        )

        # drop unnecessary columns
        columns = get_column_names(dataset)
        cols_to_remove = [c for c in columns if c not in ["wiki.pages", "question.idx"]]
        return dataset.remove_columns(cols_to_remove)

    def build_wiki_corpus(self, dataset: DatasetDict) -> Dataset:
        """Builds the Wikipedia Corpus based on extracted Wikipedia pages"""

        # step 1: flatten the nested list of pages
        dataset = dataset.map(
            FlattenWikiPages(),
            **self.map_kwargs,
            desc="Extracting Wikipedia pages",
        )

        # step 2: add the split column and concatenate the dataset splits
        dataset = DatasetDict(
            {
                k: dset.map(AddSplitInfo(k), **self.map_kwargs, desc="Add split info")
                for k, dset in dataset.items()
            }
        )

        dataset = concatenate_datasets([dataset[s] for s in sorted(dataset.keys())])

        # step 3.1: reduce the dataset per page name
        # I don't think we can use Datasets for that step unfortunately
        # todo: investigate why the fingerprint is not deterministic
        rich.print(f"[magenta]# fingerprint={dataset._fingerprint}")
        cache_file = self.cache_dir / f"{dataset._fingerprint}.jsonl"
        if not cache_file.exists():

            reduced_dataset = {}
            for row in track(dataset, description="Iterating through the dataset"):
                page = row["wiki.pages"]
                split = row["split"]
                qst_idx = row["question.idx"]

                if page not in reduced_dataset:
                    reduced_dataset[page] = {"split": [], "question.idx": []}

                reduced_dataset[page]["split"].append(split)
                reduced_dataset[page]["question.idx"].append(qst_idx)

            # step 3.2: save to file and load as Dataset
            # (if we load from the dataset from memory, it stays in memory, so we need to cache it)
            log.info(f"Saving the dataset to {cache_file}")
            reduced_dataset = ({"wiki.page": page, **row} for page, row in reduced_dataset.items())
            with open(cache_file, "w") as f:
                for row in reduced_dataset:
                    f.write(json.dumps(row) + "\n")

        # step 3.3: load the result of step 3
        dataset = Dataset.from_json(str(cache_file))

        # step 4.1: extract the page content.
        # NB: use num_proc to 1 to reduce memory use (not sure if that's a problem)
        args = copy(self.map_kwargs)
        args["num_proc"] = 1
        dataset = dataset.map(
            ExtractPageContent(self.wikipedia_data, self.wikipedia_index),
            **args,
            desc="Extracting Wikipedia pages",
        )

        return dataset.rename_columns(
            {
                "split": "question.split",
                "question.idx": "question.idx",
                "wiki.page": "title",
                "wiki.text": "text",
            }
        )

    def get_collate_pipe(self) -> Optional[Pipe]:
        return None
