from __future__ import annotations

from collections import Counter
from typing import Dict
from typing import List
from typing import Optional

from datasets import Dataset
from datasets import Split

from .base import Analytic


class ReportCorpusStatistics(Analytic):
    """Count the number of documents, tokens, and vocab size for a given corpus"""

    requires_columns: List[str] = ["document.text"]
    output_file_name: str = "corpus_statistics.json"
    batch_size = 1_000

    def process_dataset_split(
        self, dset: Dataset, *, split: Optional[str | Split] = None
    ) -> Dict | List:
        """
        Report on a specific split of the dataset.
        """
        documents_ids = Counter()
        n_tokens = 0
        vocab = set()
        for i in range(0, len(dset), self.batch_size):
            batch = dset[i : i + self.batch_size]
            paragraphs = batch["document.text"]
            documents_ids.update(batch["document.idx"])
            tokens = [token for doc in paragraphs for token in doc.split()]
            n_tokens += len(tokens)
            vocab |= set(tokens)

        n_documents = len(documents_ids)
        n_paragraphs = len(dset)
        return {
            "documents": n_documents,
            "max-paragraphs-per-doc": max(documents_ids.values()),
            "paragraphs-per-doc": n_paragraphs / n_documents,
            "paragraphs": n_paragraphs,
            "tokens-per-paragraph": n_tokens / n_paragraphs,
            "tokens": n_tokens,
            "vocab": len(vocab),
        }
