from copy import deepcopy
from typing import List
from typing import Optional
from typing import T
from typing import Tuple

import rich
import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.nn import Sequential


class Encoder(nn.Module):
    def __init__(self, output_size: int = 128):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, 1)
        self.norm_1 = nn.InstanceNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3, 1)
        self.norm_2 = nn.InstanceNorm2d(16)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216 // 4, output_size)
        self.norm_3 = nn.LayerNorm(output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.norm_1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.norm_2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.norm_3(x)
        x = self.dropout2(x)
        return x


def cast_to_device(y: T, device: torch.device) -> T:
    if isinstance(y, Tensor):
        y = y.to(device)
    elif isinstance(y, (list, tuple)):
        y = [y_.to(device) for y_ in y]
    elif isinstance(y, dict):
        y = {k: v.to(device) for k, v in y.items()}
    else:
        raise TypeError(f"Unexpected type {type(y)}")

    return y


def process_by_chunk(fn, x, chunksize=None, with_device: Optional[torch.device] = None, **kwargs):
    in_device = x.device
    if with_device is not None and x.device != with_device:
        x = cast_to_device(x, with_device)

    if chunksize is None:
        y = fn(x, **kwargs)
        y = cast_to_device(y, in_device)
        return y

    output = None
    for i in range(0, len(x), chunksize):
        y = fn(x[i : i + chunksize], **kwargs)
        y = cast_to_device(y, in_device)

        if output is None:
            output = y
        else:
            if isinstance(output, Tensor):
                output = torch.cat([output, y])
            elif isinstance(output, (List, Tuple)):
                assert len(y) == len(output)
                assert all(isinstance(y_i, Tensor) for y_i in y)
                output = [torch.cat([o, y_i]) for o, y_i in zip(output, y)]
            elif isinstance(output, dict):
                assert set(y.keys()) == set(output.keys())
                assert all(isinstance(y_i, Tensor) for y_i in y.values())
                output = {k: torch.cat([output[k], y[k]]) for k in output.keys()}
            else:
                raise TypeError(f"Unexpected type {type(output)}")
    return output


class ToyOptionRetriever(nn.Module):
    def __init__(
        self,
        hidden: int = 128,
        output_size: int = 16,
        max_chunksize: int = None,
        temperature: float = 10.0,
        share_backbone: bool = False,
        n_classes: int = 10,
    ):
        super().__init__()
        self.max_chunksize = max_chunksize
        if share_backbone:
            rich.print("[magenta]SHARING BACKBONE")
            self.backbone = Encoder(hidden)
            self.retriever_head = Sequential(nn.ReLU(), nn.Linear(hidden, output_size))
            self.reader_head = Sequential(nn.ReLU(), nn.Linear(hidden, output_size))
        else:
            self.backbone = lambda x: x
            self.retriever_head = nn.Sequential(
                Encoder(hidden), nn.ReLU(), nn.Linear(hidden, output_size)
            )
            self.reader_head = deepcopy(self.retriever_head)

        self.reader_log_temperature = nn.Parameter(torch.tensor(temperature).log())
        self.retriever_log_temperature = nn.Parameter(torch.tensor(temperature).log())
        self.n_classes = n_classes
        if n_classes is not None:
            self.reader_embeddings = nn.Embedding(n_classes, output_size)
            self.retriever_embeddings = nn.Embedding(n_classes, output_size)
        else:
            self.reader_embeddings = None
            self.retriever_embeddings = None

    @property
    def device(self):
        return next(iter(self.parameters())).device

    def _process_query(self, query: torch.Tensor):

        hq = process_by_chunk(self.backbone, query)
        h_retriever = self.retriever_head(hq)
        h_reader = self.reader_head(hq)

        if self.reader_embeddings is not None:
            h_reader = h_reader.unsqueeze(1) + self.reader_embeddings.weight.unsqueeze(0)
            h_retriever = h_retriever.unsqueeze(1) + self.retriever_embeddings.weight.unsqueeze(0)

        return {
            "retriever": (h_retriever / self.retriever_temperature()),
            "reader": (h_reader / self.reader_temperature()),
        }

    def process_query(self, query: torch.Tensor):
        max_chunksize = None if self.training else self.max_chunksize
        return process_by_chunk(
            self._process_query, query, chunksize=max_chunksize, with_device=self.device
        )

    def _process_knowledge(self, knowledge: torch.Tensor):
        dims = knowledge.shape[-3:]
        bs = knowledge.shape[:-3]
        knowledge = knowledge.view(-1, *dims)
        hk = self.backbone(knowledge)
        h_retriever = self.retriever_head(hk)
        h_reader = self.reader_head(hk)
        features = {"retriever": h_retriever, "reader": h_reader}
        features = {k: v.view(*bs, *v.shape[1:]) for k, v in features.items()}
        return features

    def process_knowledge(self, knowledge: torch.Tensor):
        max_chunksize = None if self.training else self.max_chunksize
        return process_by_chunk(
            self._process_knowledge, knowledge, chunksize=max_chunksize, with_device=self.device
        )

    def preprocess(self, h: Tensor, mode: str):
        assert mode in {"query", "knowledge"}
        op = {"query": self.process_query, "knowledge": self.process_knowledge}[mode]
        return op(h)

    def forward(self, *, q, d):
        assert q.shape[0] == d.shape[0]
        # process queries
        q_hs = self.preprocess(q, "query")
        # process knowledge
        d_hs = self.preprocess(d, "knowledge")

        return {
            "retriever_score": self.score(hq=q_hs["retriever"], hd=d_hs["retriever"]),
            "reader_score": self.score(hq=q_hs["reader"], hd=d_hs["reader"]),
        }

    def score(self, *, hd: Tensor, hq: Tensor) -> Tensor:
        if hd.dim() == 4:
            return torch.einsum("bmh,bmdh -> bmd", hq, hd)
        elif hd.dim() == 3:
            return torch.einsum("bmh,mdh -> bmd", hq, hd)
        elif hd.dim() == 2:
            return torch.einsum("bmh,dh -> bmd", hq, hd)
        else:
            raise ValueError(f"dim={hd.dim()} is not supported (shape={hd.shape})")

    def reader_temperature(self):
        return self.reader_log_temperature.exp()

    def retriever_temperature(self):
        return self.retriever_log_temperature.exp()
