import abc
from typing import List
from typing import Optional

from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.utils.datastruct import Batch


class ScoreTransform(Pipe):
    def __init__(
        self,
        *,
        field: str = "document",
        retrieval_score_key: str = "retrieval_score",
        match_score_key: str = "match_score",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.retrieval_score_key = f"{field}.{retrieval_score_key}"
        self.match_score_key = f"{field}.{match_score_key}"

    def _call_batch(self, batch: Batch, idx: Optional[List[int]] = None, **kwargs) -> Batch:

        if not {self.retrieval_score_key, self.match_score_key} <= set(batch.keys()):
            return {}

        new_scores = []
        for retrieval_score, relevance_score in zip(
            batch[self.retrieval_score_key], batch[self.match_score_key]
        ):
            new_score = self._compute_new_score(relevance_score, retrieval_score)
            new_scores.append(new_score)

        return {self.retrieval_score_key: new_scores}

    @abc.abstractmethod
    def _compute_new_score(self, relevance_score: float, retrieval_score: float) -> float:
        raise NotImplementedError


class MultiplyScoreByRelevance(ScoreTransform):
    def __init__(self, *args, factor: float = 2, **kwargs):
        super(MultiplyScoreByRelevance, self).__init__()
        self.factor = factor

    def _compute_new_score(self, relevance_score: float, retrieval_score: float) -> float:
        relevance_score = float(relevance_score > 0)
        new_score = relevance_score * self.factor * retrieval_score
        return new_score
