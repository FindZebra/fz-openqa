from itertools import chain

import numpy as np
import pandas as pd
from datasets import Dataset as HgDataset

from fz_openqa.datamodules.sampler.sampler import Sampler


class GroupedSampler(Sampler):
    """Return a list of `n_pos` and `n_neg` examples for each `question_id`"""

    idx_cols = ["question_id", "rank", "is_positive"]

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

        self.dataset: HgDataset = dataset

        # build the index
        self.index = pd.DataFrame(
            dataset.remove_columns(
                [
                    c
                    for c in self.dataset.column_names
                    if c not in self.idx_cols
                ]
            )
        ).astype(np.int32)

        # filter the index based on rank
        self.index = self.index[
            (self.index["is_positive"] > 0)
            & (self.index["rank"] <= self.pos_max_rank)
            | (self.index["is_positive"] == 0)
            & (self.index["rank"] <= self.neg_max_rank)
        ]
        self.index["_index_"] = self.index.index

        # filter out question_id with not enough positive/negative examples
        self.index["is_negative"] = self.index["is_positive"].map(
            lambda x: 1 - x
        )
        counts = self.index.groupby("question_id")[
            "is_positive", "is_negative"
        ].sum()
        counts = counts[
            (counts["is_positive"] >= n_pos) & (counts["is_negative"] >= n_neg)
        ]
        self.index = self.index[self.index["question_id"].isin(counts.index)]

        # question index
        self.q_index = list(self.index["question_id"].unique())

    def __len__(self):
        return len(self.q_index)

    def __getitem__(self, idx):
        # filter by `question_id` and `is_positive`
        subset = self.index[self.index["question_id"] == self.q_index[idx]]
        positive = subset[subset["is_positive"] > 0]
        negative = subset[subset["is_positive"] == 0]

        # get positive indexes
        positive = (
            positive["_index_"].sample(n=self.n_pos).values
            if self.n_pos > -1
            else positive["_index_"].values
        )

        # get negative indexes
        if len(negative):
            negative = (
                negative.sample(n=self.n_neg)["_index_"].values
                if self.n_neg > -1
                else negative["_index_"].values
            )
        else:
            negative = []

        # check output and return examples
        assert len(positive) == self.n_pos if self.n_pos > -1 else True
        assert len(negative) == self.n_neg if self.n_neg > -1 else True
        return [self.dataset[int(idx)] for idx in chain(positive, negative)]
