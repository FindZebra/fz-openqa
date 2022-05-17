from __future__ import annotations

import time
from typing import List
from typing import Optional
from typing import T
from typing import Tuple
from typing import Union

import rich
import torch
from loguru import logger
from torch import Tensor

from fz_openqa.datamodules.index.engines.base import IndexEngine
from fz_openqa.datamodules.index.engines.maxsim_utils.base_worker import ctx
from fz_openqa.datamodules.index.engines.maxsim_utils.base_worker import format_device
from fz_openqa.datamodules.index.engines.maxsim_utils.base_worker import WorkerSignal
from fz_openqa.datamodules.index.engines.maxsim_utils.datastruct import MaxSimInput
from fz_openqa.datamodules.index.engines.maxsim_utils.datastruct import MaxSimOutput
from fz_openqa.datamodules.index.engines.maxsim_utils.ranker import MaxSimRanker
from fz_openqa.datamodules.index.engines.maxsim_utils.reduce import MaxSimReducer
from fz_openqa.datamodules.index.engines.maxsim_utils.token_index import TokenIndex
from fz_openqa.datamodules.index.engines.maxsim_utils.utils import get_unique_pids
from fz_openqa.datamodules.index.engines.maxsim_utils.workers import MaxSimWorker
from fz_openqa.utils.datastruct import PathLike
from fz_openqa.utils.metric_type import MetricType
from fz_openqa.utils.tensor_arrow import TensorArrowTable


class MaxSim(torch.nn.Module):
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
        vectors: TensorArrowTable | Tensor,
        devices: List[int] = None,
        max_chunksize: Optional[int] = 1_000,
        max_queue_size: int = 5,
        metric_type: MetricType = MetricType.inner_product,
    ):
        super(MaxSim, self).__init__()
        self.metric_type = metric_type

        if devices is None:
            devices = list(range(torch.cuda.device_count()))
        if len(devices) == 0:
            devices = [-1]
        self.devices = [format_device(d) for d in devices]

        # Define the vectors partition
        partition = torch.linspace(0, len(vectors), len(devices) + 1, dtype=torch.long)
        partition = torch.cat([partition[:-1, None], partition[1:, None]], dim=1)
        self.register_buffer("partition", partition)

        # Initialize Input and Output queues
        self.ranking_input_queues: List[ctx.Queue] = []
        self.ranking_output_queues: List[ctx.Queue] = []
        for _ in devices:
            q = ctx.Queue(maxsize=max_queue_size)
            self.ranking_input_queues.append(q)
            q = ctx.Queue(maxsize=max_queue_size)
            self.ranking_output_queues.append(q)

        # Initialize the receiver (one iter for each Worker)
        self.receivers = [iter(q.get, WorkerSignal.EXIT) for q in self.ranking_output_queues]

        # initialize the MaxSim workers
        self.maxsim_workers = self._init_maxsim_rankers(vectors, self.devices, max_chunksize)

        # initialize the MaxSimReducer
        self.maxsim_reducer = MaxSimReducer(device=None)

    @staticmethod
    def _validate_emb2pid(emb2pid, vectors):
        if set(range(len(vectors))) != set(emb2pid.unique().tolist()):
            raise ValueError(
                f"All positions in `vector` must be in `emb2pid`."
                f"{set(range(len(vectors)))} != {set(emb2pid.unique().tolist())}"
            )

        if emb2pid.min() != 0:
            raise ValueError(f"`emb2pid` must start at 0. Found {emb2pid.min()}")

        if emb2pid.max() != len(vectors) - 1:
            raise ValueError(f"`emb2pid` must end at {len(vectors) - 1}. Found {emb2pid.max()}")

    def _init_maxsim_rankers(self, vectors, devices, max_chunksize) -> List[MaxSimWorker]:

        maxsim_workers: List[MaxSimWorker] = []
        for idx, idevice in enumerate(self.devices):
            part = self.partition[idx]

            # initialize the `MaxSimRanker` given the partition
            worker = self._init_maxsim_worker(
                vectors,
                part,
                max_chunksize=max_chunksize,
                metric_type=self.metric_type,
                id=idx,
                device=devices[idx],
                input_queue=self.ranking_input_queues[idx],
                output_queue=self.ranking_output_queues[idx],
                daemon=True,
            )
            maxsim_workers.append(worker)

        # test the consistency of the partition
        n_vecs_workers = sum(len(w.max_sim.vectors) for w in maxsim_workers)
        if len(vectors) != n_vecs_workers:
            raise ValueError(
                f"The partition is not consistent with the vectors. "
                f"len(vectors)={len(vectors)}, "
                f"workers={[len(w.max_sim.vectors) for w in maxsim_workers]}, "
                f"sum={n_vecs_workers}"
            )
        if (vectors[-1].data != maxsim_workers[-1].max_sim.vectors[-1].cpu().data).all():
            raise ValueError(
                "The partition is not consistent with the vectors. "
                "The last vector does not match the last vector of the last worker."
            )

        # start the workers
        for w in maxsim_workers:
            w.start()

        return maxsim_workers

    def _init_maxsim_worker(
        self,
        vectors: Tensor | TensorArrowTable,
        part: Tuple[int, int],
        max_chunksize: Optional[int] = None,
        metric_type: MetricType = None,
        **kwargs,
    ):
        ranker = MaxSimRanker(
            vectors, boundaries=part, max_chunksize=max_chunksize, metric_type=metric_type
        )
        worker = MaxSimWorker(max_sim=ranker, **kwargs)
        return worker

    def cpu(self: T) -> T:
        try:
            # super(MaxSim, self).cpu()
            pass
        except Exception:
            pass
        self.token_index.cpu()
        self._send_signal(WorkerSignal.TO_CPU, parallel=False)
        return self

    def to(self: T, device: torch.device) -> T:
        raise NotImplementedError(f"{self.__class__.__name__} does not support to(device)")

    def cuda(self: T, device: Optional[Union[int, torch.device]] = None) -> T:
        assert device is None
        try:
            # super(MaxSim, self).cuda()
            pass
        except Exception:
            pass
        self._send_signal(WorkerSignal.TO_CUDA, parallel=False)
        return self

    def _send_signal(self, signal: WorkerSignal, parallel=True):

        if parallel:
            for q in self.ranking_input_queues:
                q.put(signal)

            while True:
                data = [next(q, WorkerSignal.EXIT) for q in self.receivers]
                if all(d == signal for d in data):
                    break

            logger.info(f"{signal} acknowledged by all workers")

        else:
            for i, q in enumerate(self.ranking_input_queues):
                logger.info(f"Sending {signal} to worker {i}")
                q.put(signal)

                rcv = self.receivers[i]
                while True:
                    output_signal = next(rcv, WorkerSignal.EXIT)
                    if output_signal == signal:
                        logger.debug(f"{signal} acknowledged by worker {i}")
                        break

    def _collect_worker_outputs(self):
        return [next(q, WorkerSignal.EXIT) for q in self.receivers]

    def _process_batch(
        self,
        q_vectors: Tensor,
        *,
        k: int,
        pids: Tensor,
    ) -> MaxSimOutput:

        # replace nans in q_vectors (padding)
        q_vectors[q_vectors.isnan()] = 0

        # send the token_ids to each device
        _time = time.time()
        for q, device in zip(self.ranking_input_queues, self.devices):
            q_vectors_i = q_vectors.to(device, non_blocking=True)
            pids_i = pids.clone().to(device, non_blocking=True)
            worker_input = MaxSimInput(q_vectors=q_vectors_i, pids=pids_i, k=k)
            q.put(worker_input)

        # wait for the results to be ready
        maxsim_outputs = self._collect_worker_outputs()
        ranking_time = time.time() - _time

        # reduce the results
        _time = time.time()
        output: MaxSimOutput = self.maxsim_reducer(maxsim_outputs, k=k)
        reduce_time = time.time() - _time

        logger.debug(
            f"Runtime: "
            f"ranking:{ranking_time:.1f}s, "
            f"reduce:{reduce_time:.1f} | "
            f"Stats: "
            f"k={k}, input_pids: {pids.shape}, "
            f"unique_pids:{output.stats.prop_unique:.2%}, "
            f"n_docs/point: {output.stats.number_docs_per_row.mean():.1f}, "
            f"(min={output.stats.number_docs_per_row.min():.0f}, "
            f"max={output.stats.number_docs_per_row.max():.0f}), "
        )

        if torch.isnan(output.scores).any():
            rich.print(f"> scores: {output.scores}")
            raise ValueError("MaxSim returned NaNs.")

        return output

    @torch.no_grad()
    def __call__(
        self,
        q_vectors: Tensor | WorkerSignal,
        *,
        k: int = None,
        p: int = None,
        pids: Tensor = None,
        **kwargs,
    ) -> Optional[MaxSimOutput]:
        """Send data to the MaxSim pipeline"""
        if isinstance(q_vectors, torch.Tensor):
            assert k is not None, "k must be specified"
            assert pids is not None, "pids must be specified"
            return self._process_batch(q_vectors, k=k, pids=pids)

        elif isinstance(q_vectors, WorkerSignal):
            # send the signal through the pipeline
            self._send_signal(q_vectors)
        else:
            raise TypeError(f"Unsupported type: {type(q_vectors)}")

    def __del__(self):
        self.terminate()

    @property
    def all_workers(self):
        return self.maxsim_workers

    def terminate(self):
        self._send_signal(WorkerSignal.EXIT)
        for worker in self.all_workers:
            worker.join()
        for p in self.all_workers:
            p.terminate()
