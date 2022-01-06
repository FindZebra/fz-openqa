from __future__ import annotations

import logging
from typing import List
from typing import Optional
from typing import T
from typing import Tuple
from typing import Union

import numpy as np
import rich
import torch
from torch import LongTensor
from torch import nn
from torch import Tensor
from torch.multiprocessing import Queue
from torch.nn import functional as F

from fz_openqa.utils.datastruct import Batch

ctx = torch.multiprocessing.get_context("spawn")
logger = logging.getLogger(__name__)


class MaxSimPartition(nn.Module):
    """Compute MaxSim for a subset of vectors"""

    def __init__(self, vectors: Tensor, boundaries: Tensor, max_chunksize: Optional[int] = None):
        super(MaxSimPartition, self).__init__()
        self.register_buffer("vectors", vectors)
        log_mem_size(self.vectors, "MaxSimPartition vectors", logger=logger)
        self.max_chunksize = max_chunksize
        assert boundaries.shape == torch.Size([2])
        self.register_buffer("boundaries", boundaries)
        assert len(self.vectors) == boundaries[1] - boundaries[0]

    @property
    def device(self) -> torch.device:
        return self.vectors.device

    @torch.no_grad()
    def forward(
        self, q_vectors: Tensor, pids: Tensor, k: Optional[int] = None
    ) -> Tuple[Tensor, Tensor]:
        """Compute the max similarity for a batch of query vectors and
        a partition of document vectors"""

        if not q_vectors.device == self.device:
            q_vectors = q_vectors.to(self.device)
        if not pids.device == self.device:
            pids = pids.to(self.device)

        # apply the offset and set all pids that are out of range to -1
        pids -= self.boundaries[0]
        pids[(pids < 0) | (pids >= len(self.vectors))] = -1

        # get the unique pids
        pids = self._get_unique_pids(pids)

        # compute the scores
        scores = torch.zeros(
            q_vectors.shape[0], pids.shape[1], dtype=q_vectors.dtype, device=q_vectors.device
        )

        if self.max_chunksize is not None:
            chunksize = max(1, self.max_chunksize // q_vectors.shape[0])
        else:
            chunksize = pids.shape[1]
        for i in range(0, pids.shape[1], chunksize):
            scores[:, i : i + chunksize] = self._score(
                pids[:, i : i + chunksize], q_vectors, self.vectors
            )

        # if k is specified, return the top k
        if k is not None and k < scores.shape[1]:
            _, maxsim_idx = torch.topk(scores, k=k, dim=-1, largest=True, sorted=True)

            scores = scores.gather(index=maxsim_idx, dim=1)
            pids = pids.gather(index=maxsim_idx, dim=1)

        return scores, pids

    @staticmethod
    def _get_unique_pids(pids: Tensor, fill_value=-1) -> Tensor:
        """
        Get the unique pids across dimension 1 and pad to the max length.
        `torch.unique` sorts the pids, we use descending sort `* -1` to
        place negative numbers last."""
        upids = [-torch.unique(r) for r in torch.unbind(-pids)]
        max_length = max(len(r) for r in upids)

        def _pad(r):
            return F.pad(r, (0, max_length - len(r)), value=fill_value)

        return torch.cat([_pad(p)[None] for p in upids])

    @staticmethod
    def _score(pids: LongTensor, q_vectors: Tensor, vectors: Tensor):
        d_vectors = vectors[pids]

        # apply max sim to the retrieved vectors
        scores = torch.einsum("bqh, bkdh -> bkqd", q_vectors, d_vectors)
        # max. over the documents tokens, for each query token
        scores, _ = scores.max(axis=-1)
        # avg over all query tokens
        scores = scores.mean(axis=-1)
        # set the score to -inf for the negative pids
        scores[pids < 0] = -torch.inf

        return scores

    def __del__(self):
        del self.vectors
        del self.boundaries


class MaxSimWorker(ctx.Process):
    """This class Allows using a `MaxSimPartition` as a `Worker` """

    EXIT_SIGNAL = "EXIT_WORKER"
    TO_CPU_SIGNAL = "TO_CPU"
    TO_CUDA_SIGNAL = "TO_CUDA"
    COMMAND_COMPLETED_SIGNAL = "COMMAND_COMPLETED"
    BATCH_END_SIGNAL = "BATCH_END"

    def __init__(
        self,
        max_sim: MaxSimPartition,
        id: int,
        device: torch.device,
        input_queue: Queue,
        output_queue: Queue,
        *args,
        **kwargs,
    ):
        super(MaxSimWorker, self).__init__(*args, **kwargs)

        self.id = id
        self.max_sim = max_sim
        self.device = device
        self.input_queue = input_queue
        self.output_queue = output_queue

    def run(self):
        logger.debug(f"Worker {self.device} started")
        # do some initialization here
        self.max_sim = self.max_sim.to(self.device)
        logger.debug(f"Worker {self.max_sim.device} initialized")

        for data in iter(self.input_queue.get, MaxSimWorker.EXIT_SIGNAL):
            # process signals
            if isinstance(data, str):
                if data == MaxSimWorker.TO_CPU_SIGNAL:
                    self.max_sim = self.max_sim.cpu()
                    logger.debug(f"Worker {self.max_sim.device} moved to CPU")
                    self.output_queue.put(MaxSimWorker.COMMAND_COMPLETED_SIGNAL)
                elif data == MaxSimWorker.TO_CUDA_SIGNAL:
                    self.max_sim = self.max_sim.to(self.device)
                    logger.debug(f"Worker {self.max_sim.device} moved to GPU")
                    self.output_queue.put(MaxSimWorker.COMMAND_COMPLETED_SIGNAL)
                elif data == MaxSimWorker.BATCH_END_SIGNAL:
                    self.output_queue.put(MaxSimWorker.BATCH_END_SIGNAL)
                else:
                    raise ValueError(f"Unknown command {data}")

            else:
                # process data
                q_vectors, pids, k = data
                output = self.max_sim(q_vectors, pids, k)
                self.output_queue.put(output)
                del q_vectors
                del pids

        logger.debug(f"Completed: {self.device}")
        del self.max_sim


class MaxSimParallel(nn.Module):
    def __init__(
        self,
        vectors: Tensor,
        emb2pid: Tensor,
        device_ids: List[int],
        max_chunksize: Optional[int] = None,
    ):
        super(MaxSimParallel, self).__init__()
        if len(device_ids) == 0:
            device_ids = [-1]
        n_docs = emb2pid.max()
        self.register_buffer("emb2pid", emb2pid)
        self.devices = [MaxSimParallel.format_device(d) for d in device_ids]

        # define the vector partition
        partition = torch.linspace(
            0, n_docs + 1, len(device_ids) + 1, dtype=torch.long, device="cpu"
        )
        partition = torch.cat([partition[:-1, None], partition[1:, None]], dim=1)

        # initialize the `MaxSimPartition
        max_sims = []
        for part, device in zip(partition, self.devices):
            vectors_part = vectors[part[0] : part[1]]
            max_sim_part = MaxSimPartition(
                vectors_part, boundaries=part, max_chunksize=max_chunksize
            )
            max_sims.append(max_sim_part)

        assert len(vectors) == sum(len(m.vectors) for m in max_sims)
        assert (vectors[-1].data == max_sims[-1].vectors[-1].cpu().data).all()

        # initialize for multiprocessing
        self.queues = []
        self.workers = []
        self.rcv_queue = ctx.Queue()
        for i, (m, device) in enumerate(zip(max_sims, self.devices)):
            q = ctx.Queue()
            self.queues.append(q)
            # m.share_memory()
            worker = MaxSimWorker(m, i, device, q, self.rcv_queue, daemon=True)
            self.workers.append(worker)
            worker.start()

    def cpu(self: T) -> T:
        super().cpu()
        for q in self.queues:
            q.put(MaxSimWorker.TO_CPU_SIGNAL)

        self._wait_until_acknoledged()
        return self

    def cuda(self: T, device: Optional[Union[int, torch.device]] = None) -> T:
        assert device is None
        self.to(self.devices[0])
        for q in self.queues:
            q.put(MaxSimWorker.TO_CUDA_SIGNAL)

        self._wait_until_acknoledged()
        return self

    def _wait_until_acknoledged(self):
        n_acknowledged = 0
        while n_acknowledged > len(self.workers):
            output = self.rcv_queue.get()
            if output == MaxSimWorker.COMMAND_COMPLETED_SIGNAL:
                n_acknowledged += 1

    @staticmethod
    def format_device(idx: int) -> torch.device:
        if idx < 0:
            return torch.device("cpu")
        else:
            return torch.device(f"cuda:{idx}")

    @torch.no_grad()
    def forward(self, batch: Batch = None, *, k: int, chunk_size: int = 1000) -> Batch:
        if batch is None:
            return {}
        q_vectors: Tensor = batch["q_vectors"]
        topk_indices: Tensor = batch["topk_indices"]

        # 1. ge the document ids
        pids = self.emb2pid[topk_indices]

        # 2. score the pids
        # 2.a send the input data to each worker
        for q, device in zip(self.queues, self.devices):
            q.put((q_vectors.to(device, non_blocking=True), pids.to(device, non_blocking=True), k))

        # 2.b send the batch end signal
        for q in self.queues:
            q.put(MaxSimWorker.BATCH_END_SIGNAL)

        # 2.c collect the results
        outputs = []
        sum_completed = 0
        while sum_completed < len(self.queues):
            out = self.rcv_queue.get()
            if isinstance(out, str) and out == MaxSimWorker.BATCH_END_SIGNAL:
                sum_completed += 1
            else:
                out = [x.to(q_vectors.device, non_blocking=True) for x in out]
                outputs.append(out)

        scores, pids = zip(*outputs)
        scores, pids = (torch.cat(x, dim=-1) for x in (scores, pids))

        # 3. take the top-k results given the MaxSim score
        k_ = min(k, scores.shape[-1])
        scores = scores.to(torch.float32)
        _, maxsim_idx = torch.topk(scores, k=k_, dim=-1, largest=True, sorted=True)

        # 4. fetch the corresponding document indices and return
        maxsim_scores = scores.gather(index=maxsim_idx, dim=1)
        maxsim_pids = pids.gather(index=maxsim_idx, dim=1)

        if maxsim_scores.shape[1] < k or maxsim_pids.shape[1] < k:
            maxsim_pids, maxsim_scores = self._pad_outputs(k, maxsim_pids, maxsim_scores)

        # padded
        return {"pids": maxsim_pids, "scores": maxsim_scores}

    @staticmethod
    def _pad_outputs(k: int, maxsim_pids: Tensor, maxsim_scores: Tensor):
        # pad maxsim_scores with nans
        maxsim_scores = MaxSimParallel._pad_to_length(maxsim_scores, k, -torch.inf)
        # pad maxsim_pids with zeros
        maxsim_pids = MaxSimParallel._pad_to_length(maxsim_pids, k, -1)
        return maxsim_pids, maxsim_scores

    @staticmethod
    def _pad_to_length(values: Tensor, k: int, fill_value=torch.nan):
        if values.shape[1] < k:
            return F.pad(values, (0, k - values.shape[1]), value=fill_value)
        else:
            return values

    def __del__(self):
        if hasattr(self, "workers") and self.workers is not None:
            self._terminate_workers()

        del self.emb2pid

    def _terminate_workers(self):
        for q in self.queues:
            q.put(MaxSimWorker.EXIT_SIGNAL)
        for worker in self.workers:
            worker.join()
        for p in self.workers:
            p.terminate()


def log_mem_size(x, msg, logger=None):
    if isinstance(x, torch.Tensor):
        mem_size = x.element_size() * x.nelement()
    elif isinstance(x, np.ndarray):
        mem_size = x.dtype.itemsize * x.size
    else:
        raise TypeError(f"Unsupported type {type(x)}")
    mem_size /= 1024 ** 2
    prec = "MB"
    if mem_size > 1024:
        mem_size /= 1024
        prec = "GB"
    msg = f"{msg} mem. size={mem_size:.3f} {prec}, shape={x.shape}, dtype={x.dtype}"
    if logger is None:
        rich.print(msg)
    else:
        logger.info(msg)
