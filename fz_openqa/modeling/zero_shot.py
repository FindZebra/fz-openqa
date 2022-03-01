from typing import Any
from typing import Optional

import pytorch_lightning as pl
import torch
from torch import nn
from torch import tensor
from transformers import AutoModel

from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.pretty import pprint_batch


class ZeroShot(pl.LightningModule):
    """A simple BERT model for testing in a zero-shot setting."""

    def __init__(
        self,
        bert_id: str = "dmis-lab/biobert-base-cased-v1.2",
        head: str = "flat",
        limit_size: Optional[int] = None,
        **kwargs,
    ):
        super(ZeroShot, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_id)
        assert head in {"flat", "contextual"}
        self.head = head
        self.limit_size = limit_size
        self.is_colbert = nn.Parameter(tensor(head == "contextual"), requires_grad=False)
        self._limit_size = nn.Parameter(tensor(limit_size or 0), requires_grad=False)

    def forward(self, batch: Batch, **kwargs) -> Any:
        output = {}
        key_map = {"document": "_hd_", "question": "_hq_"}
        for prefix in ["document", "question"]:
            if any(str(k).startswith(prefix) for k in batch.keys()):
                input_ids = batch[f"{prefix}.input_ids"]
                attention_mask = batch[f"{prefix}.attention_mask"]
                shape = input_ids.shape
                input_ids = input_ids.view(-1, shape[-1])
                attention_mask = attention_mask.view(-1, shape[-1])
                h = self.bert(input_ids, attention_mask).last_hidden_state

                if self.head == "flat":
                    vec = h[:, 0, :]
                elif self.head == "contextual":

                    if self.limit_size is not None:
                        a = h[..., : self.limit_size - 1]
                        b = h[..., self.limit_size - 1 :].mean(-1, keepdim=True)
                        h = torch.cat([a, b], dim=-1)
                    vec = h

                else:
                    raise NotImplementedError

                vec = torch.nn.functional.standardize(vec, dim=-1)

                output_key = key_map[prefix]
                output[output_key] = vec.view(*shape[:-1], *vec.shape[1:])

        return output
