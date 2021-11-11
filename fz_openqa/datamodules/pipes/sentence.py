from collections import defaultdict
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from .base import Pipe
from fz_openqa.utils.datastruct import Batch


class GenerateSentences(Pipe):
    """A pipe to Extract sentences from a corpus of text.

    replication of:
    https://github.com/jind11/MedQA/blob/master/IR/scripts/insert_text_to_elasticsearch.py
    """

    _allows_update = False

    def __init__(
        self, *, delimiter: Optional[str] = ".", required_keys: Optional[List[str]] = None, **kwargs
    ):
        super(GenerateSentences, self).__init__(**kwargs)
        required_keys = required_keys or ["idx", "text"]
        self.delimiter = delimiter
        self.required_keys = required_keys

    def _call_batch(self, batch: Batch, **kwargs) -> Batch:
        return self.generate_sentences(
            batch, required_keys=self.required_keys, delimiter=self.delimiter
        )

    @staticmethod
    def generate_sentences(
        examples: Dict[str, List[Any]], *, required_keys: List[str], delimiter: str
    ) -> Batch:
        """
        This functions generates the sentences for each corpus article.

        return:
            - output: Batch of data (`document.text` + `idx` (document id))
        """
        # print(examples.keys())
        assert all(key in examples.keys() for key in required_keys)

        output = defaultdict(list)
        for idx, text in zip(examples["idx"], examples["text"]):
            for sent_idx, sentence in enumerate(text.split(delimiter)):
                output["idx"].append(idx)
                output["sentence_idx"].append(sent_idx)
                output["text"].append(sentence)

        return output
