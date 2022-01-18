from __future__ import annotations

import abc
import dataclasses
import logging
from enum import Enum
from typing import Any
from typing import List
from typing import Optional

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

ctx = torch.multiprocessing.get_context("spawn")
import multiprocessing.queues as mpq


def format_device(idx: None | int | torch.device) -> Optional[torch.device]:
    if idx is None:
        return None
    elif isinstance(idx, torch.device):
        return idx
    if idx < 0:
        return torch.device("cpu")
    else:
        return torch.device(f"cuda:{idx}")


class WorkerSignal(Enum):
    EXIT = "::EXIT_WORKER::"
    TO_CPU = "::TO_CPU::"
    TO_CUDA = "::TO_CUDA::"
    BATCH_END = "::BATCH_END::"
    PRINT = "::PRINT::"


class DeviceQueue(mpq.Queue):
    def __init__(self, *args, device: Optional[int | torch.device], **kwargs):
        super().__init__(*args, ctx=ctx, **kwargs)
        self._device = format_device(device)

    @property
    def device(self):
        return self._device

    @staticmethod
    def to_device(x: list | Tensor | Any | None, device):
        if x is None:
            return x
        elif isinstance(x, (list, tuple)):
            return [DeviceQueue.to_device(xi, device) for xi in x]
        elif isinstance(x, Tensor):
            return x.to(device, non_blocking=False)
        elif dataclasses.is_dataclass(x):
            sate = x.__dict__
            state = {k: DeviceQueue.to_device(v, device) for k, v in sate.items()}
            return dataclasses.replace(x, **state)
        else:
            return x

    def put(self, item: WorkerSignal | Tensor | List[Tensor], *args, **kwargs):
        item = DeviceQueue.to_device(item, self.device)
        super().put(item)

    def __getstate__(self):
        state = super().__getstate__()  # type: ignore
        return (*state, self._device)

    def __setstate__(self, state):
        *state, self._device = state
        super().__setstate__(state)  # type: ignore


class TensorWorker(ctx.Process):
    """This class allows processing Tensors using multiprocessing."""

    def __init__(
        self,
        *,
        device: torch.device | List[torch.device | int],
        id: Optional[int] = None,
        input_queue: ctx.Queue,
        output_queue: ctx.Queue | List[ctx.Queue],
        **kwargs,
    ):
        super(TensorWorker, self).__init__(**kwargs)

        self.id = id
        self.device = device
        self.input_queue = input_queue
        if not isinstance(output_queue, list):
            output_queue = [output_queue]
        self.output_queues: List[ctx.Queue] = output_queue

    def prepare(self):
        pass

    @abc.abstractmethod
    def cuda(self):
        ...

    @abc.abstractmethod
    def cpu(self):
        ...

    @abc.abstractmethod
    def process_data(self, data):
        ...

    def cleanup(self):
        pass

    def output(self, data):
        for queue in self.output_queues:
            queue.put(data)

    def run(self):
        self._log("started")
        self.prepare()
        self.cuda()
        self._log("initialized")

        with torch.no_grad():
            for data in iter(self.input_queue.get, WorkerSignal.EXIT):
                # process signals
                if isinstance(data, WorkerSignal):
                    if data == WorkerSignal.TO_CPU:
                        self._log("to cpu")
                        self.cpu()
                        self.output(WorkerSignal.TO_CPU)
                    elif data == WorkerSignal.TO_CUDA:
                        import rich

                        self._log("to cuda")
                        self.cuda()
                        self.output(WorkerSignal.TO_CUDA)
                    elif data == WorkerSignal.BATCH_END:
                        self.output(WorkerSignal.BATCH_END)
                    elif data == WorkerSignal.PRINT:
                        self._print(data)
                        self.output(WorkerSignal.PRINT)
                    else:
                        raise ValueError(f"Unknown command {data}")

                else:
                    # process data
                    self.process_data(data)

            self.output(WorkerSignal.EXIT)
            self._log("exited")
            self.cleanup()

    def _print(self, data):
        import rich

        rich.print(f"---- {type(self).__name__}(id={self.id}): {data}")

    def _log(self, status):
        logger.debug(f"{type(self).__name__} ({self.device}): {status}")


class TensorReducerWorker(TensorWorker):
    """This class allows reducing data coming from multiple workers."""

    def __init__(
        self,
        *,
        device: torch.device | List[torch.device | int],
        id: Optional[int] = None,
        input_queue: ctx.Queue | List[ctx.Queue],
        output_queue: ctx.Queue | List[ctx.Queue],
        **kwargs,
    ):
        super(TensorWorker, self).__init__(**kwargs)

        self.id = id
        self.device = device
        if not isinstance(input_queue, list):
            input_queue = [input_queue]
        self.input_queue: List[ctx.Queue] = input_queue
        if not isinstance(output_queue, list):
            output_queue = [output_queue]
        self.output_queues: List[ctx.Queue] = output_queue

    def run(self):
        self._log("started")
        self.prepare()
        self.cuda()
        self._log("initialized")

        with torch.no_grad():
            input_queue_iters = [iter(q.get, WorkerSignal.EXIT) for q in self.input_queue]
            while True:
                data = [next(q, WorkerSignal.EXIT) for q in input_queue_iters]
                if all(isinstance(d, WorkerSignal) for d in data):
                    assert all(d == data[0] for d in data)
                    data = data[0]

                # process signals
                if isinstance(data, WorkerSignal):
                    if data == WorkerSignal.TO_CPU:
                        self.cpu()
                        self._log("to cpu")
                        self.output(WorkerSignal.TO_CPU)
                    elif data == WorkerSignal.TO_CUDA:
                        self.cuda()
                        self._log("to cuda")
                        self.output(WorkerSignal.TO_CUDA)
                    elif data == WorkerSignal.BATCH_END:
                        self.output(WorkerSignal.BATCH_END)
                    elif data == WorkerSignal.PRINT:
                        import rich

                        self._print(data)
                        self.output(WorkerSignal.PRINT)
                    elif data == WorkerSignal.EXIT:
                        break
                    else:
                        raise ValueError(f"Unknown command {data}")

                else:
                    reduced_data = self.process_data(data)
                    self.output(reduced_data)

            self.output(WorkerSignal.EXIT)
            self._log("exited")
            self.cleanup()
