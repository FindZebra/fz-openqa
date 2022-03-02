from copy import deepcopy

import rich
import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor


class Encoder(nn.Module):
    def __init__(self, output_size: int = 128, max_chunksize: int = None):
        super(Encoder, self).__init__()
        self.max_chunksize = max_chunksize
        self.conv1 = nn.Conv2d(1, 8, 3, 1)
        self.conv2 = nn.Conv2d(8, 16, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216 // 4, output_size)

    def _forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        return x

    def forward(self, x):
        if self.max_chunksize is None:
            return self._forward(x)

        output = None
        for i in range(0, len(x), self.max_chunksize):
            y = self._forward(x[i : i + self.max_chunksize])
            if output is None:
                output = y
            else:
                output = torch.cat([output, y])
        return output


class ToyOptionRetriever(nn.Module):
    def __init__(
        self,
        hidden: int = 128,
        output_size: int = 16,
        max_chunksize: int = None,
        temperature: float = 10.0,
        share_backbone: bool = False,
    ):
        super().__init__()
        if share_backbone:
            rich.print("[magenta]SHARING BACKBONE")
            self.backbone = Encoder(hidden, max_chunksize=max_chunksize)
            self.retriever_head = nn.Linear(hidden, output_size)
            self.reader_head = nn.Linear(hidden, output_size)
        else:
            self.backbone = lambda x: x
            self.retriever_head = nn.Sequential(
                Encoder(hidden, max_chunksize=max_chunksize), nn.Linear(hidden, output_size)
            )
            self.reader_head = deepcopy(self.retriever_head)

        self.reader_log_temperature = nn.Parameter(torch.tensor(temperature).log())
        self.retriever_log_temperature = nn.Parameter(torch.tensor(temperature).log())

    @property
    def device(self):
        return next(iter(self.parameters())).device

    def process_query(self, query: torch.Tensor):
        hq = self.backbone(query)
        return {"retriever": self.retriever_head(hq), "reader": self.reader_head(hq)}

    def process_knowledge(self, knowledge: torch.Tensor):
        dims = knowledge.shape[-3:]
        bs = knowledge.shape[:-3]
        knowledge = knowledge.view(-1, *dims)
        hk = self.backbone(knowledge)
        features = {"retriever": self.retriever_head(hk), "reader": self.reader_head(hk)}
        features = {k: v.view(*bs, *v.shape[1:]) for k, v in features.items()}
        return features

    def forward(self, h: Tensor, mode: str):
        assert mode in {"query", "knowledge"}
        op = {"query": self.process_query, "knowledge": self.process_knowledge}[mode]
        return op(h)

    def compute_score(self, *, q, d):
        assert q.shape[0] == d.shape[0]

        # process queries
        hq = self.forward(q, "query")

        # process knowledge
        hd = self.forward(d, "knowledge")

        # compute the reader and retriever scores
        retriever_score = torch.einsum("bh,bmdh -> bmd", hq["retriever"], hd["retriever"])
        reader_score = torch.einsum("bh,bmdh -> bmd", hq["reader"], hd["reader"])

        return {
            "retriever_score": retriever_score / self.retriever_temperature(),
            "reader_score": reader_score / self.reader_temperature(),
        }

    def reader_temperature(self):
        return self.reader_log_temperature.exp()

    def retriever_temperature(self):
        return self.retriever_log_temperature.exp()
