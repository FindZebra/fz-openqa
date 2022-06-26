from typing import List
from typing import Optional

import torch
import torch.nn.functional as F
from datasets import Split

from fz_openqa.datamodules.pipes import Flatten
from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.datamodules.pipes.control.condition import In
from fz_openqa.utils.datastruct import Batch


class Transform(Pipe):
    """
    A pipe to transform a batch of examples to a batch of examples with a different structure.
    """

    def __init__(
        self,
        target_key: str = "answer.target",
        question_id_key: str = "question.id",
        keys: List[str] = None,
        input_filter=None,
        splits: Optional[List[Split]] = None,
        **kwargs,
    ):
        keys = keys or ["question.input_ids", "question.attention_mask"]
        self.target_key = target_key
        self.question_id_key = question_id_key
        self.keys = keys
        self.splits = splits
        if input_filter is not None:
            raise ValueError("FlattenMcQuestions does not support input_filter")
        input_filter = In(self.keys + [target_key])
        super().__init__(**kwargs, input_filter=input_filter)


class TransformMcQuestions(Transform):
    """
    Transform Multiple-Choice Questions weith input shape [bs, n_opts, *]
    """

    def _gather_features(self, batch: Batch) -> (torch.Size, torch.Tensor, Batch):
        # get the tensors
        targets = batch[self.target_key]
        features = {k: v for k, v in batch.items() if k in self.keys}
        eg = [x for x in features.values() if isinstance(x, torch.Tensor)][0]
        n_options = eg.shape[1]
        for k, v in features.items():
            if isinstance(v, torch.Tensor) and not v.shape[1] == n_options:
                raise ValueError(
                    f"Attribute {k} doesn't have the right number of "
                    f"options ({n_options}), found {v.shape[1]} (shape={v.shape})"
                )

        return eg.shape[:2], targets, features


class FlattenMcQuestions(TransformMcQuestions):
    """
    Flatten Multiple-Choice Questions: [bs, n_opts, *] -> [bs * n_opts, *].
    The answer target will be binarized in the process.
    """

    @torch.no_grad()
    def _call_batch(
        self, batch: Batch, idx: Optional[List[int]] = None, split: Optional[Split] = None, **kwargs
    ) -> Batch:
        if self.splits is not None and split not in self.splits:
            return {}

        # get the features
        shape, targets, features = self._gather_features(batch)

        # build binary targets
        binary_targets = torch.zeros(*shape, dtype=torch.bool, device=targets.device)
        binary_targets.scatter_(1, targets.unsqueeze(1), 1)

        # build the `question_id` feature
        question_id = torch.arange(shape[0], dtype=torch.long, device=targets.device)
        question_id = question_id.unsqueeze(1).expand(shape).contiguous()

        # prepare the output
        output = {
            self.target_key: binary_targets,
            self.question_id_key: question_id,
            **features,
        }

        # flatten the features
        output = Flatten(level=1)(output)
        return output


class OptionDropout(TransformMcQuestions):
    """
    Drop out options during training.
    """

    @torch.no_grad()
    def _call_batch(
        self, batch: Batch, idx: Optional[List[int]] = None, split: Optional[Split] = None, **kwargs
    ) -> Batch:

        if self.splits is not None and split not in self.splits:
            return {}

        # get the tensors
        target = batch[self.target_key]
        features = {k: v for k, v in batch.items() if k in self.keys}  # todo: refactor
        eg = [x for x in features.values() if isinstance(x, torch.Tensor)][0]
        n_options = eg.shape[1]
        for k, v in features.items():
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
        for key, x in features.items():
            if isinstance(x, torch.Tensor):
                leaf_shape = x.shape[len(selected_ids.shape) :]
                _index = selected_ids.view(*selected_ids.shape, *(1 for _ in leaf_shape))
                _index = _index.expand(*selected_ids.shape, *leaf_shape)
                output[key] = x.gather(dim=1, index=_index)
            else:
                output[key] = [[y[i] for i in ids] for y, ids in zip(x, selected_ids)]

        # pprint_batch(output, "OptionDropout")
        return output


class OptionBinarized(OptionDropout):
    @torch.no_grad()
    def _call_batch(
        self, batch: Batch, idx: Optional[List[int]] = None, split: Optional[Split] = None, **kwargs
    ) -> Batch:

        if self.splits is not None and split not in self.splits:
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

        # sample label
        rdn_idx = F.gumbel_softmax(logits, hard=False)
        rdn_idx = rdn_idx.argmax(dim=1)
        answer_target = rdn_idx == target
        selected_ids = rdn_idx[:, None]

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

        return output
