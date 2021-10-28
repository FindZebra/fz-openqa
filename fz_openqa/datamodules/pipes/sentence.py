from collections import defaultdict
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.utils.datastruct import Batch


class GenerateSentences(Pipe):
    """A pipe to Extract sentences from a corpus of text.

    replication of:
    https://github.com/jind11/MedQA/blob/master/IR/scripts/insert_text_to_elasticsearch.py
    """

    def __init__(
        self,
        *,
        delimitter: Optional[str] = ".",
        required_keys: Optional[Sequence[str]] = ["idx", "text"]
    ):
        self.delimitter = delimitter
        self.required_keys = required_keys

    def __call__(self, batch: Batch) -> Batch:
        output = self.generate_sentences(
            batch,
            keys=self.required_keys,
        )

        return output

    def generate_sentences(
        self,
        examples: Dict[str, List[Any]],
        keys: List[str],
    ) -> Tuple[List[int], Batch]:
        """
        This functions generates the sentences for each corpus article.

        return:
            - output: Batch of data (`document.text` + `idx` (document id))
        """
        # print(examples.keys())
        assert all(key in examples.keys() for key in self.required_keys)

        output = defaultdict(list)
        for idx, text in zip(examples["idx"], examples["text"]):
            for sent_idx, sentence in enumerate(text.split(self.delimitter)):
                output["idx"].append(idx)
                output["sentence_idx"].append(sent_idx)
                output["text"].append(sentence)

        return output
