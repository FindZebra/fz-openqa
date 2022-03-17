from collections import defaultdict
from typing import Dict

import torch

from fz_openqa.datamodules.pipes import PrioritySampler
from fz_openqa.toys.model import ToyOptionRetriever


def weighted_mc_sample(logits, k: int, n: int = 100):
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
        sampler: str = "priority",
    ):
        self.s_range = s_range
        self.data = data
        self.targets = targets
        self.knowledge = knowledge
        self.scores = {k: None for k in data.keys()}
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.sampler = {"priority": PrioritySampler.sample, "weighted_mc": weighted_mc_sample}[
            sampler
        ]

    @torch.no_grad()
    def index(self, model: ToyOptionRetriever = None):
        if model is None:
            m, nd, *_ = self.knowledge.shape
            for split, data in self.data.items():
                nq = data.shape[0]
                self.scores[split] = torch.zeros((nq, m, nd), device=self.device)
        else:
            h_knowledge = model.process_knowledge(self.knowledge)["retriever"]
            for split, data in self.data.items():
                self.scores[split] = self._compute_score_for_split(model, data, h_knowledge)

    def _compute_score_for_split(
        self, model: ToyOptionRetriever, data: torch.Tensor, h_knowledge: torch.Tensor
    ):
        h_dataset = model.process_query(data)["retriever"]
        scores = torch.einsum("qh,mdh -> qmd", h_dataset, h_knowledge)
        # normalize scores
        s_max = scores.max(dim=-1, keepdim=True).values
        s_min = scores.min(dim=-1, keepdim=True).values
        scores = (scores - s_min) / (s_max - s_min)
        scores = self.s_range * scores
        return scores

    def __call__(self, qids: torch.Tensor, split: str, k: int = 10):
        assert self.scores is not None

        # slice scores
        q_scores = self.scores[split][qids]

        # priority sample
        z, log_weight = self.sampler(q_scores, k)

        # slice the scores
        q_scores = q_scores.gather(2, index=z)

        # slice the corresponding evidences
        knowledge = self.knowledge.unsqueeze(0)
        m = len(z.shape)
        z = z.view(*z.shape, *(1 for _ in knowledge.shape[m:]))
        z = z.expand(*z.shape[:m], *knowledge.shape[m:])
        knowledge = knowledge.expand(z.shape[0], *knowledge.shape[1:])
        evidences = knowledge.gather(2, index=z)
        return evidences, q_scores, log_weight

    @property
    def device(self):
        return next(iter(self.data.values())).device

    def iter_split(self, split: str):
        indexes = torch.randperm(self.data[split].shape[0], device=self.device)
        for i in range(0, self.data[split].shape[0], self.batch_size):
            qids = indexes[i : i + self.batch_size]
            yield self.get(qids, split)

    def get(self, idx, split):
        is_int = False
        if isinstance(idx, int):
            idx = slice(idx, idx + 1)
            is_int = True

        evidences, q_scores, log_weight = self.__call__(idx, split, k=self.n_samples)
        data = {
            "question": self.data[split][idx],
            "target": self.targets[split][idx],
            "evidence": evidences,
            "retrieval_score": q_scores,
            "retrieval_log_weight": log_weight,
        }
        if is_int:
            data = {k: v.squeeze(0) for k, v in data.items()}

        return data