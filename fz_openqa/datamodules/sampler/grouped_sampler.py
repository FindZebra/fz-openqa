from itertools import chain

import numpy as np
import pandas as pd
import rich
from datasets import Dataset as HgDataset

from fz_openqa.datamodules.sampler.sampler import Sampler


class GroupedSampler(Sampler):
    """Sample both `n_neg` negative and `n_pos` positive examples"""

    idx_cols = ["idx", "question_id", "rank", "is_positive"]

    def __init__(
        self,
        *,
        dataset: HgDataset,
        n_pos: int = 1,
        n_neg: int = 1,
        pos_max_rank: int = 0,
        neg_max_rank: int = 10,
    ):
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.pos_max_rank = pos_max_rank
        self.neg_max_rank = neg_max_rank
        assert all(
            k in dataset.column_names for k in self.idx_cols
        ), f"{dataset.column_names}"
        self.q_idxes = [int(x) for x in dataset["question_id"]]
        self.dset: HgDataset = dataset
        self.index = pd.DataFrame(
            dataset.remove_columns(
                [c for c in self.dset.column_names if c not in self.idx_cols]
            )
        ).astype(np.int32)
        self.index["_index_"] = self.index.index

    def __len__(self):
        return len(self.q_idxes)

    def __getitem__(self, idx):
        subset = self.index[self.index["question_id"] == self.q_idxes[idx]]
        positive = subset[
            (subset["is_positive"] > 0) & (subset["rank"] <= self.pos_max_rank)
        ]
        negative = subset[
            (subset["is_positive"] == 0)
            & (subset["rank"] <= self.neg_max_rank)
        ]
        positive = positive["_index_"].sample(n=self.n_pos).values
        negative = (
            negative.sample(n=self.n_neg)["_index_"].values
            if len(negative)
            else []
        )
        assert len(positive) == self.n_pos
        assert len(negative) == self.n_neg
        return [self.dset[int(idx)] for idx in chain(positive, negative)]
