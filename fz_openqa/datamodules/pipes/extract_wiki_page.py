import itertools
from typing import Dict
from typing import List

import rich

from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.datamodules.utils.typing import HfDataset
from fz_openqa.utils.datastruct import Batch


class ExtractWikiPage(Pipe):
    def __init__(self, *, wiki_data: HfDataset, wiki_index: Dict, query_key: str):
        # Wikipedia dump to extract page content
        self.wiki_data = wiki_data
        # Index to look up Wikipedia pages and extract page content
        self.wiki_index = wiki_index
        self.query_key = query_key

    def extract_content(self, pages: List[str]) -> List[Dict]:
        """ Takes a list of Wiki-page titles and extracts the page content based on a Wiki-dump. """
        wiki_content = []
        for page_title in pages:
            # Look up index in Wikipedia dump og page title
            wiki_idx = self.wiki_index.get(page_title)
            # Get page content in dump
            wiki_page = self.wiki_data.__getitem__(wiki_idx)
            # Append page title and page content to an output list
            wiki_content.append({"title": page_title, "content": wiki_page["text"]})
        return wiki_content

    def __call__(self, batch: Batch, query_key: str = "wiki.pages", **kwargs) -> Batch:
        query_key = query_key or self.query_key
        assert query_key is not None, "attribute `text_key` must be set."
        batch["wiki.text"] = [self.extract_content(eg) for eg in batch[query_key]]
        return batch
