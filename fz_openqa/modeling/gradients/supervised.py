import torch
from torch import Tensor
from torch.nn import functional as F

from fz_openqa.modeling.gradients.base import Gradients


class SupervisedGradients(Gradients):
    def __call__(self, partial_score: Tensor, match_score: Tensor, **kwargs):
        """
        Compute the supervised retrieval loss
        # todo: check loss, can we keep it without using the mask
        # todo: figure out how to compute the targets and logits for the metrics
        """

        pos_docs = match_score > 0
        loss_mask = pos_docs.sum(-1) > 0
        logits = partial_score[loss_mask]
        pos_docs = pos_docs[loss_mask].float()

        if logits.numel() > 0:
            n_total = len(pos_docs)
            n_pos = pos_docs.sum()
            loss = -(pos_docs * F.log_softmax(logits, dim=-1) / pos_docs.sum(dim=-1, keepdims=True))
            loss = loss.sum(-1)

        else:
            n_total = n_pos = 0
            loss = torch.tensor(0.0, dtype=partial_score.dtype, device=partial_score.device)

        # compute logits and targets for the metrics
        match_score = match_score[loss_mask]
        ids = torch.argsort(match_score, dim=-1, descending=True)
        targets = torch.zeros((logits.shape[0],), dtype=torch.long, device=logits.device)
        logits = logits.gather(index=ids, dim=-1)

        return {
            "retriever/loss": loss,
            "_retriever_logits_": logits,
            "_retriever_targets_": targets,
            "retriever/n_options": n_total,
            "retriever/n_positive": n_pos,
        }
