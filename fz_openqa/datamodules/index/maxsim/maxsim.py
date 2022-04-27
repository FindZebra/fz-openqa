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

from fz_openqa.datamodules.index.handlers.base import IndexHandler
from fz_openqa.datamodules.index.maxsim.base_worker import ctx
from fz_openqa.datamodules.index.maxsim.base_worker import format_device
from fz_openqa.datamodules.index.maxsim.base_worker import WorkerSignal
from fz_openqa.datamodules.index.maxsim.datastruct import MaxSimInput
from fz_openqa.datamodules.index.maxsim.datastruct import MaxSimOutput
from fz_openqa.datamodules.index.maxsim.ranker import MaxSimRanker
from fz_openqa.datamodules.index.maxsim.reduce import MaxSimReducer
from fz_openqa.datamodules.index.maxsim.token_index import TokenIndex
from fz_openqa.datamodules.index.maxsim.utils import get_unique_pids
from fz_openqa.datamodules.index.maxsim.workers import MaxSimWorker
from fz_openqa.utils.datastruct import PathLike
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
        token_index: IndexHandler | PathLike,
        vectors: TensorArrowTable | Tensor,
        emb2pid: TensorArrowTable | Tensor,
        ranking_devices: List[int],
        faiss_devices: List[int],
        max_chunksize: Optional[int] = 10_000,
        max_queue_size: int = 5,
        deduplicate_pids: bool = True,
    ):
        super(MaxSim, self).__init__()
        logger.info(f"Setting MaxSim with ranking devices: {ranking_devices}")
        logger.info(f"Setting MaxSim with faiss (tokens) devices: {faiss_devices}")

        self.deduplicate_pids = deduplicate_pids
        # init the token_index
        self.token_index = TokenIndex(token_index, faiss_devices)

        # Store `emb2pid`
        self._validate_emb2pid(emb2pid, vectors)
        # Add `-1` : padding
        ones = torch.ones_like(emb2pid[..., -1:])
        emb2pid = torch.cat([emb2pid, -ones], dim=-1)
        self.register_buffer("emb2pid", emb2pid)

        # setup the Rankers
        # init the devices
        if len(ranking_devices) == 0:
            ranking_devices = [-1]
        self.ranking_devices = [format_device(d) for d in ranking_devices]

        # Define the vectors partition
        partition = torch.linspace(0, emb2pid.max() + 1, len(ranking_devices) + 1, dtype=torch.long)
        partition = torch.cat([partition[:-1, None], partition[1:, None]], dim=1)
        self.register_buffer("partition", partition)

        # Initialize Input and Output queues
        self.ranking_input_queues: List[ctx.Queue] = []
        self.ranking_output_queues: List[ctx.Queue] = []
        for _ in ranking_devices:
            q = ctx.Queue(maxsize=max_queue_size)
            self.ranking_input_queues.append(q)
            q = ctx.Queue(maxsize=max_queue_size)
            self.ranking_output_queues.append(q)

        # Initialize the receiver (one iter for each Worker)
        self.receivers = [iter(q.get, WorkerSignal.EXIT) for q in self.ranking_output_queues]

        # initialize the MaxSim workers
        self.maxsim_workers = self._init_maxsim_rankers(
            vectors, self.ranking_devices, max_chunksize
        )

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
        for idx, idevice in enumerate(self.ranking_devices):
            part = self.partition[idx]

            # initialize the `MaxSimRanker` given the partition
            worker = self._init_maxsim_worker(
                vectors,
                part,
                max_chunksize=max_chunksize,
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
        **kwargs,
    ):
        ranker = MaxSimRanker(vectors, boundaries=part, max_chunksize=max_chunksize)
        worker = MaxSimWorker(max_sim=ranker, **kwargs)
        return worker

    def cpu(self: T) -> T:
        try:
            # super(MaxSim, self).cpu()
            pass
        except Exception:
            pass
        self.token_index.cpu()
        self._send_signal(WorkerSignal.TO_CPU)
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
        self.token_index.cuda()
        self._send_signal(WorkerSignal.TO_CUDA)
        return self

    def _send_signal(self, signal: WorkerSignal):

        for q in self.ranking_input_queues:
            q.put(signal)

        while True:
            data = [next(q, WorkerSignal.EXIT) for q in self.receivers]
            if all(d == signal for d in data):
                break

        logger.info(f"{signal} acknowledged by all workers")

    def _collect_worker_outputs(self):
        return [next(q, WorkerSignal.EXIT) for q in self.receivers]

    def _process_batch(
        self, q_vectors: Tensor, *, p: int, k: int, doc_ids: Optional[List[int]] = None
    ) -> MaxSimOutput:

        # replace nans in q_vectors (padding)
        q_vectors[q_vectors != q_vectors] = 0

        # process q_vectors using the token-level faiss index
        _time = time.time()
        token_ids = self.token_index(q_vectors, p, doc_ids=doc_ids)
        faiss_time = time.time() - _time

        # retriever the pids from the token_ids, at this step, there are duplicated pids.
        pids = self.emb2pid[token_ids.to(self.emb2pid.device)]

        # Deduplicate pids; this step is done on `device`, which is set by default on CPU.
        # This step can be quite slow, but it saves a lot of memory.
        if self.deduplicate_pids:
            _time = time.time()
            pids = get_unique_pids(pids)
            deduplicate_time = time.time() - _time
        else:
            deduplicate_time = 0

        # send the token_ids to each device
        _time = time.time()
        for q, device in zip(self.ranking_input_queues, self.ranking_devices):
            q_vectors_i = q_vectors.to(device, non_blocking=True)
            pids_i = pids.clone().to(device, non_blocking=True)
            worker_input = MaxSimInput(q_vectors=q_vectors_i, pids=pids_i, k=k)
            q.put(worker_input)

        # wait for the results to be ready
        maxsim_outputs = self._collect_worker_outputs()
        ranking_time = time.time() - _time

        # reduce the results
        _time = time.time()
        output = self.maxsim_reducer(maxsim_outputs, k=k)
        reduce_time = time.time() - _time

        logger.info(
            f"Runtime: "
            f"faiss:{faiss_time:.1f}s, "
            f"deduplicate:{deduplicate_time:.1f}s, "
            f"ranking:{ranking_time:.1f}s, "
            f"reduce:{reduce_time:.1f}s"
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
        doc_ids: Optional[List[int]] = None,
        **kwargs,
    ) -> Optional[MaxSimOutput]:
        """Send data to the MaxSim pipeline"""
        if isinstance(q_vectors, torch.Tensor):
            assert p is not None and k is not None, "p and k must be specified"
            return self._process_batch(q_vectors, p=p, k=k, doc_ids=doc_ids)

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
