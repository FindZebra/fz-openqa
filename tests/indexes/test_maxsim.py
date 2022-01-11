from unittest import TestCase

import faiss
import torch

from fz_openqa.datamodules.index.utils.io import build_emb2pid_from_vectors
from fz_openqa.datamodules.index.utils.maxsim.base_worker import WorkerSignal
from fz_openqa.datamodules.index.utils.maxsim.maxsim import MaxSim
from fz_openqa.datamodules.index.utils.maxsim.ranker import MaxSimOutput


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
        self.maxsim.put(WorkerSignal.PRINT, k=None, p=None)
        self.maxsim.cuda()
        self.maxsim.cpu()

    @torch.no_grad()
    def test_search(self):
        n_trials = 3
        pids = torch.tensor([13, 14, 15, 67, 68, 69])
        input_vectors = self.vectors[pids]
        for _ in range(n_trials):
            self.maxsim.put(input_vectors, k=10, p=100)

        for x in self.maxsim.get():
            assert isinstance(x, MaxSimOutput)
            assert (x.pids[:, 0] == pids).all()

    def tearDown(self) -> None:
        self.maxsim.terminate()
