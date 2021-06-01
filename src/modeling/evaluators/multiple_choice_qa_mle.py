import collections

from datasets import Split
from torch.nn import functional as F
from torchmetrics.classification.accuracy import Accuracy

from .abstract import *


class MultipleChoiceQaMaximumLikelihood(Evaluator):
    """
    Evaluates the Reader model `p(a_i | q, e, A)` using maximum likelihood estimation
    in a multiple choice QA context (A = [a_1,...a_P]). The loss is defined as:

        L =  \sum_p log p(a_p | q, e, A) 1(p = a)

    where a is the index of the true answer.
    """
    # TODO: formalize the Metrics logic (when to compute and log)
    text_key = "question"
    _required_eval_feature_names = [
        'answer_idx',
    ]

    def __init__(self):
        super().__init__()
        self.accuracy = nn.ModuleDict(
            {f"_{k}": Accuracy() for k in [Split.TRAIN,
                                           Split.VALIDATION,
                                           Split.TEST]})

    def forward(self, model: nn.Module, batch: Any, split: str, **kwargs: Any) -> Dict[str, Tensor]:
        self.check_batch_type(batch)
        self.check_feature_names(batch)

        logits: Tensor = model(batch)
        targets: Tensor = batch['answer_idx']
        loss = F.cross_entropy(logits, targets, reduce="mean")
        acc = self.accuracy[f"_{split}"](logits.argmax(-1), targets)
        return {'loss': loss, 'accuracy': acc}

    def check_feature_names(self, batch):
        for f in self._required_eval_feature_names:
            assert f in batch.keys(), f"The feature {f} is required for evaluation."

    def check_batch_type(self, batch):
        assert isinstance(
            batch,
            (
                dict,
                collections.OrderedDict,
                collections.UserDict,
            ),
        )
