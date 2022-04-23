from __future__ import annotations

import torch

from fz_openqa.datamodules.index.maxsim.base_worker import format_device
from fz_openqa.datamodules.index.maxsim.base_worker import TensorWorker
from fz_openqa.datamodules.index.maxsim.datastruct import MaxSimInput
from fz_openqa.datamodules.index.maxsim.datastruct import MaxSimOutput
from fz_openqa.datamodules.index.maxsim.ranker import MaxSimRanker


class MaxSimWorker(TensorWorker):
    """This class allows computing MaxSim for a subset of vectors (contained in MaxSimRanker)."""

    def __init__(
        self,
        max_sim: MaxSimRanker,
        id: int,
        **kwargs,
    ):
        super(MaxSimWorker, self).__init__(**kwargs)

        self.id = id
        self.max_sim = max_sim

    def _print(self, data):
        import rich  # noqa: F811

        rich.print(
            f"---- {type(self).__name__}(id={self.id}): {data}, device={self.max_sim.device}"
        )

    def cuda(self):
        assert isinstance(self.device, (int, torch.device))
        device = format_device(self.device)
        self.max_sim = self.max_sim.to(device, non_blocking=False)

    def cpu(self):
        self.max_sim = self.max_sim.to(torch.device("cpu"), non_blocking=False)

    def process_data(self, data):
        # rich.print(f"> {type(self).__name__}(id={self.id})")
        assert isinstance(data, MaxSimInput)
        scores, pids = self.max_sim(data.q_vectors, data.pids, data.k)
        output = MaxSimOutput(
            scores=scores, pids=pids, k=data.k, boundaries=self.max_sim.boundaries
        )
        self.output(output)

    def cleanup(self):
        del self.max_sim
