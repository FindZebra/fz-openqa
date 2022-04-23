from unittest import TestCase

import faiss
import rich
import torch

from fz_openqa.datamodules.index.utils.io import build_emb2pid_from_vectors
from fz_openqa.datamodules.index.maxsim import WorkerSignal
from fz_openqa.datamodules.index.maxsim.maxsim import MaxSim
from fz_openqa.datamodules.index.maxsim.datastruct import MaxSimOutput


class TestMaxSim(TestCase):

    def setUp(self) -> None:

        # Generate random vectors
        vdim = 32
        seq_len = 100
        self.vectors = torch.randn(size=(100, seq_len, vdim))
        emd2pid = build_emb2pid_from_vectors(self.vectors)

        # build index
        index = faiss.IndexFlatL2(self.vectors.shape[-1])
        index.add(self.vectors.view(-1, self.vectors.shape[-1]))

        # initialize MaxSim
        self.maxsim = MaxSim(token_index=index,
                             vectors=self.vectors,
                             emb2pid=emd2pid,
                             ranking_devices=[-1, -1, -1],
                             faiss_devices=[-1])

    def test_signals(self):
        self.maxsim(WorkerSignal.PRINT)
        # self.maxsim.cuda()
        self.maxsim.cpu()

    @torch.no_grad()
    def test_search(self):
        n_trials = 3
        pids = torch.tensor([13, 14, 15, 67, 68, 69])
        input_vectors = self.vectors[pids]
        for _ in range(n_trials):
            x = self.maxsim(input_vectors, k=10, p=100)
            assert isinstance(x, MaxSimOutput)
            if not (x.pids[:, 0] == pids).all():
                rich.print("=== target pids[:, 0] ===")
                rich.print(pids)
                rich.print("=== pids ===")
                rich.print(x.pids[:, :10])
                raise AssertionError("pids are not equal")

    def tearDown(self) -> None:
        self.maxsim.terminate()
