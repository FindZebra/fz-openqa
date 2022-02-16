from __future__ import annotations

from typing import Any
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional

import numpy as np
import torch
from datasets import Dataset
from datasets import Split
from torch import Tensor

from ..utils.dataset import keep_only_columns
from .base import Analytic


def safe_cast_to_list(x: Any) -> Any:
    if isinstance(x, list):
        return [safe_cast_to_list(y) for y in x]
    elif isinstance(x, Tensor):
        if x.dim() == 0:
            return x.item()
        else:
            return [safe_cast_to_list(y) for y in x]


class SequenceLengths(Analytic):
    """Report the di"""

    requires_columns: List[str] = []
    output_file_name = "sequence_lengths.json"
    batch_size = 1000

    def process_dataset_split(
        self, dset: Dataset, *, split: Optional[str | Split] = None
    ) -> Dict | List:
        """
        Report on a specific split of the dataset.
        """
        output = {}

        for field in ["document", "question"]:
            key = f"{field}.attention_mask"
            if key not in dset.column_names:
                continue

            # drop unused columns
            dset = keep_only_columns(dset, [key])

            lengths = []
            for i in range(0, len(dset), self.batch_size):
                seqs = dset[i : i + self.batch_size]
                seqs = seqs[key]
                for seq in self.yield_sequences(seqs):
                    lengths += [len([y for y in seq if y > 0])]

            output[field] = {
                "min": min(lengths),
                "max": max(lengths),
                "mean": np.mean(lengths),
                "p75": np.percentile(lengths, 75),
                "p95": np.percentile(lengths, 95),
            }

        return output

    @staticmethod
    def yield_sequences(x) -> Iterator[List[int]]:

        # cast values to lists
        if isinstance(x, Tensor):
            x = x.cpu().numpy()
        if isinstance(x, (np.ndarray)):
            x = x.tolist()

        # iterate and yield
        if isinstance(x[0], int):
            yield x
        elif isinstance(x[0], (list, np.ndarray, torch.Tensor)):
            for y in x:
                yield from SequenceLengths.yield_sequences(y)
        else:
            raise ValueError(f"Unsupported sequence type: {type(x[0])}")
