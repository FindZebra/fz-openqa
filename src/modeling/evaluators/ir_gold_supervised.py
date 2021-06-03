from datasets import Split
from torch.nn import functional as F
from torchmetrics import Metric
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy

from .abstract import *
from src.modeling.similarities import Similarity

class InformationRetrievalGoldSupervised(Evaluator):
    """
    Evaluates the Reader model `p(a_i | q, e, A)` using maximum likelihood estimation
    in a multiple choice QA context (A = [a_1,...a_P]). The loss is defined as:

        L =  \sum_p log p(a_p | q, e, A) 1(p = a)

    where a is the index of the true answer.
    """
    # TODO: formalize the Metrics logic (when to compute and log)
    _required_eval_feature_names = [
        'is_gold',
        'question.input_ids',
        'question.attention_mask',
        'question.input_ids',
        'document.attention_mask',
        'question.input_ids',
        'question.attention_mask',
        'is_gold'
    ]

    def __init__(self, similarity:Similarity):
        super().__init__()
        self.similarity = similarity
        gen_metric = lambda split: MetricCollection([Accuracy()]
                                                    , prefix=f"{split}/")
        self.metrics = nn.ModuleDict(
            {f"_{split}": gen_metric(split) for split in [Split.TRAIN,
                                                          Split.VALIDATION,
                                                          Split.TEST]})

    def get_metric(self, split: str) -> Metric:
        return self.metrics[f"_{split}"]

    def forward(self, model: nn.Module, batch: Any, split: str, **kwargs: Any) -> Dict[str, Tensor]:
        self.check_batch_type(batch)
        self.check_feature_names(batch)

        hq = model(input_ids=batch['question.input_ids'],
                   attention_mask=batch['question.attention_mask'],
                   key='document')
        he = model(input_ids=batch['document.input_ids'],
                   attention_mask=batch['document.attention_mask'],
                   key='document')

        logits = self.similarity(hq, he)
        targets = torch.arange(start=0, end=len(logits)).long().to(logits.device)
        loss = F.cross_entropy(logits, targets)
        self.get_metric(split).update(logits.argmax(-1), targets)
        return {'loss': loss}

    def check_feature_names(self, batch):
        for f in self._required_eval_feature_names:
            assert f in batch.keys(), f"The feature {f} is required for evaluation."

    def reset_metrics(self, split: Optional[str] = None) -> None:
        """reset the metrics"""
        if split is None:
            map(lambda m: m.reset(), self.metrics.values())
        else:
            self.get_metric(split).reset()

    def compute_metrics(self, split: Optional[str] = None) -> Dict[str, Tensor]:
        """Compute the metrics"""
        if split is not None:
            metrics = [self.get_metric(split)]
        else:
            metrics = self.metrics.values()

        output = {}
        for metric in metrics:
            output.update(**metric.compute())

        return output
