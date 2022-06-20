import time
from abc import ABCMeta
from typing import Callable
from typing import List
from typing import Union

import wikipedia
from loguru import logger

from fz_openqa.utils.datastruct import Batch


class QueryWikiAPI:
    __metaclass__ = ABCMeta

    def __init__(self, text_key: str, n_results: int = 10):
        self.text_key = text_key
        self.n_results = n_results

    def query_api(self, answer_str: str) -> List[str]:
        """Returns a list of all the article's titles (max 10) that contain the query."""
        while True:
            try:
                output = wikipedia.search(answer_str, results=self.n_results)
                break
            except Exception as exc:
                logger.warning(exc)
                answer_str = answer_str[:299]
                time.sleep(1)

        return output

    @staticmethod
    def _extract_wiki_pages(answer_options: Union[str, List], fn: Callable) -> List[str]:
        """Takes a string or list of strings, queries each string against the Wikipedia API."""
        if isinstance(answer_options, str):
            return fn(answer_options)
        elif isinstance(answer_options, (tuple, list)):
            page_lst = [fn(option) for option in answer_options]
            flatten_lst = [page for sublist in page_lst for page in sublist]
            return flatten_lst
        else:
            ValueError(f"Cannot handle type {type(answer_options).__name__}.")

    def __call__(self, batch: Batch, text_key: str = "answer.text", **kwargs) -> Batch:
        text_key = text_key or self.text_key
        assert text_key is not None, "attribute `text_key` must be set."
        batch["wiki.query"] = [
            self._extract_wiki_pages(eg, self.query_api) for eg in batch[text_key]
        ]
        return batch
