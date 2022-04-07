from collections import defaultdict
from collections import namedtuple
from typing import Dict

import rich
import torch
from torch.utils.data import Dataset

from fz_openqa.datamodules.pipes import PrioritySampler
from fz_openqa.toys.model import process_by_chunk
from fz_openqa.toys.model import ToyOptionRetriever

Sampled = namedtuple("Sampled", "pid score log_weight")


def weighted_mc_sample(logits, *, k: int, n: int = 100):
    device = logits.device
    probs = torch.softmax(logits, dim=-1)
    *bs, n_choices = probs.shape
    probs = probs.view(-1, n_choices)

    zs = []
    log_weights = []
    MAX_ITERS = 1e6

    for probs_k in probs:
        n_ = n
        z_counts = defaultdict(lambda: 0)
        j = 0
        while len(z_counts.keys()) < k:
            # sample
            z = torch.multinomial(probs_k, n_, replacement=True)

            # take the top-k values
            zu, zc = z.unique(return_counts=True)
            idx = torch.argsort(zc, descending=True)[:k]
            zu = zu[idx]
            zc = zc[idx]

            # update the counts
            for zu_, zc_ in zip(zu, zc):
                z_counts[zu_] += zc_

            j += 1
            if j > MAX_ITERS:
                raise RuntimeError("Too many iterations")

            if j % 10 == 0:
                n_ *= 2

        z_counts = list(sorted(z_counts.items(), key=lambda x: x[1], reverse=True))[:k]
        z = torch.tensor([x for x, c in z_counts], dtype=torch.long, device=device)
        zc = torch.tensor([c for x, c in z_counts], dtype=logits.dtype, device=device)
        zs.append(z)
        log_weights.append(torch.log(zc / zc.sum()))

    zs = torch.stack(zs, dim=0)
    log_weights = torch.stack(log_weights, dim=0)

    return zs.view(*bs, k), log_weights.view(*bs, k)


logits = 5 * torch.randn(2, 3, 1000)
z, log_w = weighted_mc_sample(logits, k=10)


class ToySampler:
    def __init__(
        self,
        data: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        knowledge: torch.Tensor,
        s_range=100,
        batch_size: int = 32,
        n_samples: int = 10,
        n_classes: int = None,
        chunksize: int = 1024,
        sampler: str = "priority",
        device: str = "cpu",
        sample_device: str = "cpu",
        sample_chunksize: int = 10,
    ):
        self.device = torch.device(device)
        self.sample_device = torch.device(sample_device)
        self.s_range = s_range
        self.n_classes = n_classes
        self.data = data
        self.targets = targets
        self.knowledge = knowledge
        self.chunksize = chunksize
        self.sample_chunksize = sample_chunksize
        self.scores = {k: None for k in data.keys()}
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.sampler = {"priority": PrioritySampler.sample, "weighted_mc": weighted_mc_sample}[
            sampler
        ]
        self.sampled_data = {k: None for k in data.keys()}

    @torch.no_grad()
    def index(self, model: ToyOptionRetriever = None):
        if model is None:
            nd = self.knowledge.shape[0]
            for split, data in self.data.items():
                nq = data.shape[0]
                self.scores[split] = torch.zeros(
                    (nq, self.n_classes, nd), device=torch.device("cpu")
                )
        else:
            h_knowledge = model.process_knowledge(self.knowledge)["retriever"]
            for split, data in self.data.items():
                score = process_by_chunk(
                    self._compute_score_for_split,
                    data,
                    model=model,
                    h_knowledge=h_knowledge,
                    chunksize=self.chunksize,
                )
                self.scores[split] = score.to(self.device)

    def _compute_score_for_split(
        self, data: torch.Tensor, *, model: ToyOptionRetriever, h_knowledge: torch.Tensor
    ):
        h_dataset = model.process_query(data)["retriever"]
        scores = model.score(hq=h_dataset, hd=h_knowledge)
        # normalize scores
        s_max = scores.max(dim=-1, keepdim=True).values
        s_min = scores.min(dim=-1, keepdim=True).values
        scores = (scores - s_min) / (s_max - s_min)
        scores = self.s_range * scores
        return scores

    def __call__(self, qids: torch.Tensor, *, split: str, k: int = 10):
        assert self.scores is not None

        # slice scores
        q_scores = self.scores[split][qids]

        # priority sample
        z, log_weight = process_by_chunk(
            self.sampler,
            q_scores,
            k=k,
            with_device=self.sample_device,
            chunksize=self.sample_chunksize,
        )

        # slice the scores
        q_scores = q_scores.gather(2, index=z)

        return Sampled(pid=z, score=q_scores, log_weight=log_weight)

    def gather_knowledge(self, pids):
        # slice the corresponding evidences
        m = len(pids.shape)
        pids = pids.view(*pids.shape, *(1 for _ in self.knowledge.shape[1:]))
        pids = pids.expand(*pids.shape[:m], *self.knowledge.shape[1:])
        # expand the knowledge
        knowledge = self.knowledge.view(*(1 for _ in range(m - 1)), *self.knowledge.shape)
        knowledge = knowledge.expand(*pids.shape[: m - 1], *self.knowledge.shape)
        evidences = knowledge.gather(1, index=pids)
        return evidences

    def iter_split(self, split: str):
        indexes = torch.randperm(self.data[split].shape[0], device=self.device)
        for i in range(0, self.data[split].shape[0], self.batch_size):
            qids = indexes[i : i + self.batch_size]
            yield self.get(qids, split=split)

    def sample(self, *, split: str, k: int = None):
        if k is None:
            k = self.n_samples
        indexes = torch.arange(self.data[split].shape[0], device=self.device)
        pids, scores, log_weight = self.__call__(indexes, split=split, k=k)
        self.sampled_data[split] = Sampled(pid=pids, score=scores, log_weight=log_weight)

    def get(self, idx, *, split):
        assert self.sampled_data[split] is not None
        sampled = self.sampled_data[split]

        data = {
            "question": self.data[split][idx],
            "target": self.targets[split][idx],
            "evidence": self.gather_knowledge(sampled.pid[idx]),
            "retrieval_score": sampled.score[idx],
            "retrieval_log_weight": sampled.log_weight[idx],
        }

        return data


class SplitWrapper(Dataset):
    def __init__(self, sampler: ToySampler, split: str, device=None):
        sampler.sample(split=split)
        self.sampler = sampler
        self.split = split
        self.device = device

    def __len__(self):
        return len(self.sampler.data[self.split])

    def __getitem__(self, idx):
        batch = self.sampler.get(idx, split=self.split)
        if self.device is not None:
            batch = {k: v.to(self.device) for k, v in batch.items()}
        return batch
