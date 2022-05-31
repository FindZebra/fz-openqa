from __future__ import annotations

from typing import Dict
from typing import List

import numpy as np
import rich
import torch
from torchvision import datasets

DEFAULT_LABELS = [1, 3, 7, 9]


class ClassDataset(datasets.MNIST):
    def __init__(self, label: int, group: None | str, *args, noise_level: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)

        # take only the images with the given label
        mask = self.targets == label
        self.data = self.data[mask]
        self.targets = self.targets[mask]

        # slice the dataset for the given group
        assert group in {None, "data", "knowledge", "all"}
        if group is not None:
            rgn = np.random.RandomState(0)
            mask = rgn.binomial(1, 0.5, size=self.targets.shape).astype(np.bool)
            if group == "data":
                pass
            elif group == "knowledge":
                mask = ~mask
            elif group == "all":
                mask = slice(None, None)
            else:
                raise ValueError("Invalid group value")
            self.data = self.data[mask]
            self.targets = self.targets[mask]

        # preprocessing
        self.data = self.preprocess(self.data, noise_level=noise_level)

    @staticmethod
    def preprocess(data, noise_level: float = 1.0):
        data = data.unsqueeze(1)
        data = data.float()
        data = 2 * data - 1

        if noise_level > 0:
            # mask = data.clone()
            # mask.bernoulli_(1 - noise_level)
            # data *= mask.float()

            noise = torch.randn_like(data)
            mask = torch.empty((len(data),)).uniform_()
            mask = mask.view(-1, 1, 1, 1)
            perturb_level = mask.clone().uniform_(1, 2).long().float()
            data = torch.where(mask < noise_level, noise, data + perturb_level * noise)
            data = data.clamp(-1, 1)

        return data


def concatenate(datasets: Dict, labels: list[int]):
    all_data = {
        k: (v.data, labels.index(k) * torch.ones(size=v.data.shape[:1], dtype=torch.long))
        for k, v in datasets.items()
    }
    data = torch.cat([v[0] for v in all_data.values()])
    targets = torch.cat([v[1] for v in all_data.values()])
    return data, targets


def generate_toy_datasets(*, labels: List[int] = -1, noise_level: float = 0, **dataset_kwargs):
    if isinstance(labels, int) and labels == -1:
        labels = DEFAULT_LABELS

    # generate a dataset for each group and each label
    train_data = {
        i: ClassDataset(i, "data", train=True, noise_level=0, **dataset_kwargs).data for i in labels
    }
    knowledge = {
        i: ClassDataset(
            i,
            "knowledge" if i in labels else "all",
            train=True,
            noise_level=noise_level,
            **dataset_kwargs
        ).data
        for i in range(10)
    }

    knowledge_labels = {
        i: torch.full((k.shape[0],), i, dtype=torch.long) for i, k in knowledge.items()
    }

    test_data = {
        i: ClassDataset(i, None, train=False, noise_level=0, **dataset_kwargs).data for i in labels
    }

    # concatenate the train/test datasets across labels
    train_data, train_labels = concatenate(train_data, labels)
    test_data, test_labels = concatenate(test_data, labels)

    # concatenate the knowledge bases
    ids = list(knowledge.keys())
    knowledge = torch.cat([knowledge[i] for i in ids], dim=0)
    knowledge_labels = torch.cat([knowledge_labels[i] for i in ids], dim=0)

    return {
        "train": {"targets": train_labels, "data": train_data},
        "test": {"targets": test_labels, "data": test_data},
        "knowledge": knowledge,
        "knowledge_labels": knowledge_labels,
    }
