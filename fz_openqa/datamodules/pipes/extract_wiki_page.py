import itertools
from typing import Dict
from typing import List

import rich

from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.datamodules.utils.typing import HfDataset
from fz_openqa.utils.datastruct import Batch


class ExtractWikiPage(Pipe):
    def __init__(self, *, wiki_data: HfDataset, query_key: str):
        self.wiki_data = wiki_data
        self.query_key = query_key

    def extract_content(self, pages: List[str]) -> List[Dict]:
        wiki_content = []
        for page_title in pages:
            wiki_pages = self.wiki_data.filter(lambda row: row["title"] == page_title)
            if wiki_pages:
                wiki_content.append({"title": page_title, "content": wiki_pages["text"]})
        return wiki_content

    def __call__(self, batch: Batch, query_key: str = "wiki.pages", **kwargs) -> Batch:
        query_key = query_key or self.query_key
        assert query_key is not None, "attribute `text_key` must be set."
        batch["wiki.text"] = [self.extract_content(eg) for eg in batch[query_key]]
        return batch
