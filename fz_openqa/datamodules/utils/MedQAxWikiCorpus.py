import re
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional

import rich as rich
import wikipedia
from rich.progress import track
from rich.status import Status

import fz_openqa
from fz_openqa.datamodules import DataModule
from fz_openqa.datamodules.builders import MedQABuilder
from fz_openqa.datamodules.pipes import TextFormatter


class WikixMedQaCorpusBuilder:
    def __init__(self, dataset_builder, article_titles: Optional[Dict] = {}):
        # Index article titles to escape duplicates
        self.article_titles = article_titles
        # Here I want to instantiate the MedQa dataset
        # to hold Questions and Answer Options (No tokenization needed)
        self.dataset_builder = dataset_builder or MedQABuilder(
            text_formatter=TextFormatter(
                remove_hex=True,
                remove_ref=True,
                remove_breaks=True,
            ),
            cache_dir=Path(fz_openqa.__file__).parent.parent / "cache",
            num_proc=4,
        )
        self.dataset = DataModule(builder=self.dataset_builder, train_batch_size=10)

        with Status("Instantiating MedQa dataset.."):
            self.dataset.prepare_data()
            self.dataset.setup()
            rich.print(self.dataset)

        self.dataloaders = {
            "train": self.dataset.train_dataloader(),
            "val": self.dataset.val_dataloader(),
            "test": self.dataset.test_dataloader(),
        }

    @staticmethod
    def _format_content(text: str) -> str:
        text = re.sub(r"[\n\r\t]+", " ", text)
        if re.search("References", text):
            start = re.search("References", text).start()
            text = text[:start]
        text = re.sub(r"=+", " ", text)

        return text

    @staticmethod
    def _search_wiki_pages(answer_options: List[str]) -> List:
        for answer_str in answer_options:
            yield wikipedia.search(answer_str)

    def _extract_wiki_content(self, wiki_pages: List[str]) -> List[Dict]:
        for page in wiki_pages:
            if page in self.article_titles.keys():
                pass
            try:
                self.article_titles[page] = ""
                article = wikipedia.search(page)
                yield {"title": article.title, "content": self._format_content(article.content)}
            except (wikipedia.DisambiguationError, wikipedia.PageError, KeyError) as e:
                pass

    def __call__(self) -> List:

        # Json list to store Wikipedia content
        json_list = []
        for split, dataloader in self.dataloaders.items():
            for batch in track(dataloader, description=f"Iterating through the {split} dataset..."):
                wiki_searches = map(self._search_wiki_pages, batch["answer.text"])
                wiki_content = map(self._extract_wiki_content, wiki_searches)

                for eg in wiki_content:
                    json_list.append(eg)

        return json_list
