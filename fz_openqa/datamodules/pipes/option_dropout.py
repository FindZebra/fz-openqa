from typing import List
from typing import Optional

import torch
import torch.nn.functional as F
from datasets import Split

from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.datamodules.pipes.control.condition import In
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.pretty import pprint_batch


class OptionDropout(Pipe):
    """
    Drop out options during training.
    """

    def __init__(
        self,
        target_key: str = "answer.target",
        keys: List[str] = None,
        input_filter=None,
        **kwargs,
    ):
        keys = keys or ["question.input_ids", "question.attention_mask"]
        self.target_key = target_key
        self.keys = keys
        assert input_filter is None, "OptionDropout does not support input_filter"
        input_filter = In(self.keys + [target_key])
        super(OptionDropout, self).__init__(**kwargs, input_filter=input_filter)

    @torch.no_grad()
    def _call_batch(
        self, batch: Batch, idx: Optional[List[int]] = None, split: Optional[Split] = None, **kwargs
    ) -> Batch:

        if split != Split.TRAIN:
            return {}

        # get the tensors
        target = batch[self.target_key]
        values = {k: v for k, v in batch.items() if k in self.keys}  # todo: refactor
        eg = [x for x in values.values() if isinstance(x, torch.Tensor)][0]
        n_options = eg.shape[1]
        for k, v in values.items():
            if isinstance(v, torch.Tensor) and not v.shape[1] == n_options:
                raise ValueError(
                    f"Attribute {k} doesn't have the right number of "
                    f"options ({n_options}), found {v.shape[1]} (shape={v.shape})"
                )

        # define sampling probs
        logits = torch.ones(eg.shape[0], n_options, dtype=torch.float32, device=eg.device)
        logits.scatter_(dim=1, index=target.unsqueeze(1), value=-float("inf"))

        # sample labels #2:
        rdn_idx = F.gumbel_softmax(logits, hard=False)
        rdn_idx = rdn_idx.argmax(dim=1)

        # gather data
        selected_ids = torch.cat([target[:, None], rdn_idx[:, None]], dim=1)

        # generate the new answer target
        answer_target = torch.zeros_like(rdn_idx)
        answer_target.bernoulli_(0.5)

        # shuffle selected ids
        mask = torch.cat([answer_target[:, None], 1 - answer_target[:, None]], dim=1)
        selected_ids = selected_ids.gather(dim=1, index=mask)

        # for i in range(len(target)):
        #     msg = f"# logits={logits[i]}, target={target[i]}, rdn_idx={rdn_idx[i]}"
        #     assert selected_ids[i][answer_target[i]] == target[i]
        #     assert selected_ids[i, 0] != selected_ids[i, 1], msg

        # output
        output = {self.target_key: answer_target}
        for key, x in values.items():
            if isinstance(x, torch.Tensor):
                leaf_shape = x.shape[len(selected_ids.shape) :]
                _index = selected_ids.view(*selected_ids.shape, *(1 for _ in leaf_shape))
                _index = _index.expand(*selected_ids.shape, *leaf_shape)
                output[key] = x.gather(dim=1, index=_index)
            else:
                output[key] = [[y[i] for i in ids] for y, ids in zip(x, selected_ids)]

        # pprint_batch(output, "OptionDropout")
        return output
