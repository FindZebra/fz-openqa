import abc
import math
from typing import List
from typing import Optional
from typing import Tuple

from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.utils.datastruct import Batch


class ScoreTransform(Pipe):
    def __init__(
        self,
        *,
        field: str = "document",
        exponentiate: bool = False,
        max_score: Optional[float] = 1.0,
        proposal_score_key: str = "proposal_score",
        match_score_key: str = "match_score",
        temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.exponentiate = exponentiate
        self.max_score = max_score
        self.proposal_score_key = f"{field}.{proposal_score_key}"
        self.match_score_key = f"{field}.{match_score_key}"
        self.temperature = temperature

    def _call_batch(self, batch: Batch, idx: Optional[List[int]] = None, **kwargs) -> Batch:

        if not {self.proposal_score_key, self.match_score_key} <= set(batch.keys()):
            return {}

        new_proposal_scores = []
        new_match_scores = []
        proposal_scores: List[List[float]] = batch[self.proposal_score_key]
        match_scores: List[List[float]] = batch[self.match_score_key]
        for m_scores_eg, r_scores_eg in zip(match_scores, proposal_scores):
            r_scores_eg_, m_scores_eg_ = self._process_one_eg(m_scores_eg, r_scores_eg)
            new_proposal_scores.append(r_scores_eg_)
            new_match_scores.append(m_scores_eg_)

        return {
            self.proposal_score_key: new_proposal_scores,
            self.match_score_key: new_match_scores,
        }

    def _process_one_eg(
        self, match_scores: List[float], proposal_scores: List[float]
    ) -> Tuple[List[float], List[float]]:

        if self.max_score is not None:
            match_scores = [min(score, self.max_score) for score in match_scores]
        if self.exponentiate:
            match_scores = [math.exp(score) if score > 0 else 0.0 for score in match_scores]

        # compute statistics that can be used to transform the score
        stats = {
            "max_relevance_scores": max(match_scores),
            "sum_relevance_scores": sum(x for x in match_scores),
        }

        new_proposal_scores = []
        new_match_scores = []
        for proposal_score, match_score in zip(proposal_scores, match_scores):
            # transform the scores based on the match score
            new_proposal_score, new_match_score = self._compute_new_score(
                match_score, proposal_score, **stats
            )

            # apply temperature
            new_proposal_score = new_proposal_score / self.temperature

            # append results
            new_proposal_scores.append(new_proposal_score)
            new_match_scores.append(new_match_score)
        return new_proposal_scores, new_match_scores

    @abc.abstractmethod
    def _compute_new_score(
        self,
        relevance_score: float,
        proposal_score: float,
        *,
        max_relevance_scores: float,
        sum_relevance_scores: float,
    ) -> Tuple[float, float]:
        return proposal_score, relevance_score


class MultiplyScoreByRelevance(ScoreTransform):
    def __init__(self, *args, factor: float = 1, normalize: bool = True, **kwargs):
        super(MultiplyScoreByRelevance, self).__init__()
        self.factor = factor
        self.normalize = normalize

    def _compute_new_score(
        self,
        relevance_score: float,
        proposal_score: float,
        *,
        max_relevance_scores: float,
        sum_relevance_scores: float,
    ) -> Tuple[float, float]:
        multiplier = float(relevance_score)

        if multiplier == 0:
            return proposal_score, relevance_score

        if self.normalize:
            multiplier = multiplier / sum_relevance_scores

        new_score = proposal_score + self.factor * multiplier * proposal_score

        return new_score, multiplier
