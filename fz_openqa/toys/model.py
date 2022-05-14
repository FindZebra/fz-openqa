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
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 5, stride=1)
        self.norm_1 = nn.InstanceNorm2d(8)
        self.conv2 = nn.Conv2d(8, 8, 5, stride=2)
        self.norm_2 = nn.InstanceNorm2d(8)
        self.conv3 = nn.Conv2d(8, 8, 5, stride=2)
        self.dropout_conv = nn.Dropout(0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.norm_1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.norm_2(x)
        x = self.dropout_conv(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = torch.flatten(x, 1)
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

    if chunksize is None:
        if with_device is not None and x.device != with_device:
            x = cast_to_device(x, with_device)
        y = fn(x, **kwargs)
        y = cast_to_device(y, in_device)
        return y

    output = None
    for i in range(0, len(x), chunksize):
        x_chunk = x[i : i + chunksize]
        if with_device is not None and x.device != with_device:
            x_chunk = cast_to_device(x_chunk, with_device)
        y = fn(x_chunk, **kwargs)
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
        hidden = 72
        if share_backbone:
            rich.print("[magenta]SHARING BACKBONE")
            self.backbone = Encoder()
            self.retriever_head = Sequential(nn.ReLU(), nn.Linear(hidden, output_size))
            self.reader_head = Sequential(nn.ReLU(), nn.Linear(hidden, output_size))
        else:
            self.backbone = lambda x: x
            self.retriever_head = nn.Sequential(
                Encoder(), nn.ReLU(), nn.Linear(hidden, output_size)
            )
            self.reader_head = deepcopy(self.retriever_head)

        self.reader_log_temperature = nn.Parameter(torch.tensor(temperature).log())
        self.retriever_log_temperature = nn.Parameter(torch.tensor(temperature).log())
        self.n_classes = n_classes
        if n_classes is not None:
            self.reader_embeddings = nn.Embedding(n_classes, 28 ** 2)
            self.retriever_embeddings = nn.Embedding(n_classes, 28 ** 2)
        else:
            self.reader_embeddings = None
            self.retriever_embeddings = None

    @property
    def device(self):
        return next(iter(self.parameters())).device

    def get_embedding_as_image(self, key):
        emb = {
            "reader": self.reader_embeddings.weight,
            "retriever": self.retriever_embeddings.weight,
        }[key]

        return emb.view(self.n_classes, 1, 28, 28)

    def _process_query(self, query: torch.Tensor):

        hq = process_by_chunk(self.backbone, query)
        h_retriever = self.retriever_head(hq)
        h_reader = self.reader_head(hq)

        if self.reader_embeddings is not None:
            x_emb = self.get_embedding_as_image("reader")
            h_emb = self.backbone(x_emb)
            reader_emb = self.reader_head(h_emb)
            h_reader = h_reader.unsqueeze(1) + reader_emb.unsqueeze(0)

            x_emb = self.get_embedding_as_image("retriever")
            h_emb = self.backbone(x_emb)
            retriever_emb = self.retriever_head(h_emb)
            h_retriever = h_retriever.unsqueeze(1) + retriever_emb.unsqueeze(0)

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
