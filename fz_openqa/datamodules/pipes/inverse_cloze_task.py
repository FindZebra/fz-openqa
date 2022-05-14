from __future__ import annotations

from collections import defaultdict
from typing import List
from typing import Optional

import numpy as np
import torch

from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.utils.datastruct import Batch


class InverseClozeTask(Pipe):
    def __init__(
        self,
        document_key: str = "document.idx",
        passage_key: str = "document.passage_idx",
        input_field: str = "document",
        output_field: str = "question",
        score_keys: str | List[str] = None,
        max_score: float = 100.0,
        min_distance: float = 1,
        poisson_lambda: float = 1,
        n_neighbours: int = 1,
        keys: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(score_keys, str):
            score_keys = [score_keys]

        self.document_key = document_key
        self.passage_key = passage_key
        self.input_field = input_field
        self.output_field = output_field
        self.min_distance = min_distance
        self.score_keys = score_keys or ["document.match_score", "document.proposal_score"]
        self.max_score = max_score
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
        sampled_scores = []
        for i, (doc_id, passage_id) in enumerate(zip(doc_ids, passage_ids)):
            # get the list of ids matching the doc_id
            candidates_ids = doc_partition[doc_id]

            # sample distances without replacement
            samples_i = []
            scores_i = []
            while len(samples_i) < self.n_neighbours:
                # sample a distance
                d_j = np.random.poisson(self.poisson_lambda)
                d_j = max(d_j, self.min_distance)

                # sample a passage given this distance
                sign = 2 * int(np.random.random() > 0.5) - 1
                j = i + sign * d_j

                # reject if already included
                if j in samples_i:
                    continue

                # check if j is from the same document
                if j not in candidates_ids:
                    continue

                # the, if the passage is valid, append
                samples_i.append(j)
                score_j = max(1.0, self.max_score) - d_j
                scores_i.append(score_j)

            # sort descending
            data = zip(samples_i, scores_i)
            data = sorted(data, key=lambda x: -x[1])
            samples_i, scores_i = map(list, zip(*data))

            # append
            sampled_indices.append(samples_i)
            sampled_scores.append(scores_i)

        # rename the batch
        output = {
            k.replace(f"{self.input_field}.", f"{self.output_field}."): v for k, v in batch.items()
        }

        # add the "document" field
        for key in self.keys:
            values = output[f"{self.output_field}.{key}"]
            output[f"{self.input_field}.{key}"] = [
                self._concat([values[i] for i in selected_ids]) for selected_ids in sampled_indices
            ]

        # add the score column(s)
        for key in self.score_keys:
            output[key] = sampled_scores

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
