from __future__ import annotations

import logging
from typing import Iterable
from typing import List
from typing import Optional
from typing import T
from typing import Tuple
from typing import Union

import faiss
import numpy as np
import torch
from torch import Tensor

from fz_openqa.datamodules.index.utils.maxsim.base_worker import ctx
from fz_openqa.datamodules.index.utils.maxsim.base_worker import DeviceQueue
from fz_openqa.datamodules.index.utils.maxsim.base_worker import format_device
from fz_openqa.datamodules.index.utils.maxsim.base_worker import WorkerSignal
from fz_openqa.datamodules.index.utils.maxsim.ranker import MaxSimOutput
from fz_openqa.datamodules.index.utils.maxsim.ranker import MaxSimRanker
from fz_openqa.datamodules.index.utils.maxsim.workers import FaissInput
from fz_openqa.datamodules.index.utils.maxsim.workers import FaissWorker
from fz_openqa.datamodules.index.utils.maxsim.workers import MaxSimReducerWorker
from fz_openqa.datamodules.index.utils.maxsim.workers import MaxSimWorker
from fz_openqa.utils.datastruct import PathLike
from fz_openqa.utils.tensor_arrow import TensorArrowTable

logger = logging.getLogger(__name__)


class MaxSim(object):
    """Implement the two steps retrieval process of Colbert: https://arxiv.org/abs/2004.12832

    This object functions as a multiprocessing pipeline with 3 types of workers:
    * n=1 FaissWorker: contains the faiss index for the token embeddings
    * n=len(`ranking_devices`) MaxSimWorker: each contains a subset of the vectors, compute
        the MaxSim score for the token ids
    * n=1 MaxSimReducerWorker: reduce the MaxSim scores from all the MaxSimWorkers

    The data flows is:
    * (q_vectors)-> FaissWorker -(token_ids)-> N x [MaxSimWorker] -(pids)-> MaxSimReducerWorker



    """

    def __init__(
        self,
        *,
        token_index: faiss.Index | PathLike,
        vectors: TensorArrowTable | Tensor,
        emb2pid: TensorArrowTable | Tensor,
        ranking_devices: List[int],
        faiss_devices: List[int],
        max_chunksize: Optional[int] = 10_000,
        max_queue_size: int = 5,
    ):
        if len(ranking_devices) == 0:
            ranking_devices = [-1]

        reduce_device = format_device(ranking_devices[-1])  # todo

        # initialize Input and Output queues
        self.input_queue = DeviceQueue(device=-1)
        self.ranking_input_queues: List[DeviceQueue] = []
        self.ranking_output_queues: List[DeviceQueue] = []
        for device in ranking_devices:
            q = DeviceQueue(device=device, maxsize=max_queue_size)
            self.ranking_input_queues.append(q)
            q = DeviceQueue(device=None, maxsize=max_queue_size)
            self.ranking_output_queues.append(q)

        reduce_max_queue_size = max_queue_size * len(ranking_devices) if max_queue_size else None
        self.reduce_input_queue = DeviceQueue(device=reduce_device, maxsize=reduce_max_queue_size)
        self.output_queue = DeviceQueue(device=torch.device("cpu"), maxsize=max_queue_size)

        # initialize the `MaxSimReducerWorker`
        self.reducer_worker = MaxSimReducerWorker(
            device=format_device(reduce_device),
            input_queue=self.ranking_output_queues,
            output_queue=[self.output_queue],
            daemon=True,
        )
        self.reducer_worker.start()

        # initialize the MaxSim workers
        self._init_maxsim_rankers(emb2pid, vectors, ranking_devices, max_chunksize)

        # initialize the faiss index
        self.faiss_worker = FaissWorker(
            token_index,
            device=faiss_devices,
            input_queue=self.input_queue,
            output_queue=self.ranking_input_queues,
            daemon=True,
        )
        self.faiss_worker.start()

    def _init_maxsim_rankers(self, emb2pid, vectors, devices, max_chunksize):
        partition = np.linspace(0, emb2pid.max() + 1, len(devices) + 1, dtype=int)
        max_sims = []

        self.maxsim_workers: List[MaxSimWorker] = []
        for idx, i in enumerate(range((len(partition) - 1))):
            part = (partition[i], partition[i + 1])

            # initialize the `MaxSimRanker` given the partition
            worker = self._init_maxsim_worker(
                vectors,
                emb2pid,
                part,
                max_chunksize=max_chunksize,
                id=idx,
                device=devices[idx],
                input_queue=self.ranking_input_queues[idx],
                output_queue=self.ranking_output_queues[idx],
                daemon=True,
            )
            self.maxsim_workers.append(worker)

        # test the consistency of the partition
        assert len(vectors) == sum(len(w.max_sim.vectors) for w in self.maxsim_workers)
        assert (vectors[-1].data == self.maxsim_workers[-1].max_sim.vectors[-1].cpu().data).all()

        # start the workers
        for w in self.maxsim_workers:
            w.start()
        return max_sims

    def _init_maxsim_worker(
        self,
        vectors: Tensor | TensorArrowTable,
        emb2pid: Tensor | TensorArrowTable,
        part: Tuple[int, int],
        max_chunksize: Optional[int] = None,
        **kwargs,
    ):
        ranker = MaxSimRanker(emb2pid, vectors, boundaries=part, max_chunksize=max_chunksize)
        worker = MaxSimWorker(max_sim=ranker, **kwargs)
        return worker

    def cpu(self: T) -> T:
        self.input_queue.put(WorkerSignal.TO_CPU)

        self._wait_until_acknowledged(WorkerSignal.TO_CPU)
        return self

    def cuda(self: T, device: Optional[Union[int, torch.device]] = None) -> T:
        assert device is None
        self.input_queue.put(WorkerSignal.TO_CUDA)
        self._wait_until_acknowledged(WorkerSignal.TO_CUDA)
        return self

    def _wait_until_acknowledged(self, signal: WorkerSignal):
        for output in iter(self.output_queue.get, WorkerSignal.EXIT):
            if output == signal:
                break

        logger.info(f"{signal} acknowledged by all workers")

    @torch.no_grad()
    def put(self, q_vectors: Tensor | WorkerSignal, *, k: int, p: int, idx: Optional[int] = None):
        """Send data to the MaxSim pipeline"""
        if isinstance(q_vectors, torch.Tensor):
            input_data = FaissInput(q_vectors=q_vectors, k=k, p=p, idx=idx)
            self.input_queue.put(input_data)

        elif isinstance(q_vectors, WorkerSignal):
            # send the signal through the pipeline
            signal = q_vectors
            self.input_queue.put(signal)
            self._wait_until_acknowledged(signal)
        else:
            raise TypeError(f"Unsupported type: {type(q_vectors)}")

    def get(self) -> Iterable[MaxSimOutput]:
        """Gather the output data from the MaxSim pipeline"""
        self.input_queue.put(WorkerSignal.BATCH_END)
        for data in iter(self.output_queue.get, WorkerSignal.EXIT):
            if isinstance(data, MaxSimOutput):
                yield data
            elif isinstance(data, WorkerSignal):
                if data == WorkerSignal.BATCH_END:
                    break
                else:
                    raise RuntimeError(f"Unexpected signal: {data}")

    def __del__(self):
        self.terminate()

    @property
    def all_workers(self):
        return self.maxsim_workers + [self.reducer_worker, self.faiss_worker]

    def terminate(self):
        self.input_queue.put(WorkerSignal.EXIT)
        for worker in self.all_workers:
            worker.join()
        for p in self.all_workers:
            p.terminate()
