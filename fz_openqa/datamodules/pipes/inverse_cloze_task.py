from __future__ import annotations

from collections import defaultdict
from typing import List
from typing import Optional

import numpy as np
import rich
import torch

from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.pretty import pprint_batch


class InverseClozeTask(Pipe):
    def __init__(
        self,
        document_key: str = "document.idx",
        passage_key: str = "document.passage_idx",
        input_field: str = "document",
        output_field: str = "question",
        min_distance: float = 1,
        poisson_lambda: float = 1,
        n_neighbours: int = 1,
        keys: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if n_neighbours > 1:
            raise NotImplementedError("n_neighbours > 1 not implemented")

        self.document_key = document_key
        self.passage_key = passage_key
        self.input_field = input_field
        self.output_field = output_field
        self.min_distance = min_distance
        self.poisson_lambda = poisson_lambda
        self.n_neighbours = n_neighbours
        self.keys = keys

    def _call_batch(self, batch: Batch, idx: Optional[List[int]] = None, **kwargs) -> Batch:
        doc_ids = batch[self.document_key]
        passage_ids = batch[self.passage_key]

        # gather passage_ids per document_idx
        doc_partition = defaultdict(list)
        for i, doc_id in enumerate(doc_ids):
            doc_partition[doc_id].append(i)

        sampled_indices = []
        for i, (doc_id, passage_id) in enumerate(zip(doc_ids, passage_ids)):
            # get the target distance from the current passage
            distance = np.random.poisson(self.poisson_lambda)
            distance = max(self.min_distance, distance)

            # get the set of candidate documents
            candidates_ids = doc_partition[doc_id]
            candidates = [i for i in candidates_ids if abs(passage_ids[i] - passage_id) == distance]
            sampled_indices.append(np.random.choice(candidates))

        # rename the batch
        output = {
            k.replace(f"{self.input_field}.", f"{self.output_field}."): v for k, v in batch.items()
        }

        # add the "document" field
        for key in self.keys:
            values = output[f"{self.output_field}.{key}"]
            output[f"{self.input_field}.{key}"] = [
                self._concat([values[i]]) for i in sampled_indices
            ]

        output[f"{self.input_field}.match_score"] = [[1] for i in sampled_indices]

        return output

    @staticmethod
    def _concat(values: List[List | torch.Tensor]) -> List | torch.Tensor:
        if isinstance(values[0], list):
            return values
        elif isinstance(values[0], torch.Tensor):
            if len(values) == 1:
                return values[0].unsqueeze(0)
            else:
                return torch.cat([v[None] for v in values], dim=0)

        else:
            raise NotImplementedError("Unsupported type")
