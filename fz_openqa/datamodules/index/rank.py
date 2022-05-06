from collections import defaultdict
from itertools import zip_longest
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple

import datasets
from loguru import logger

from fz_openqa.datamodules.index.index_pipes import FetchDocuments
from fz_openqa.datamodules.pipes import ApplyAsFlatten
from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.datamodules.pipes import Sequential
from fz_openqa.datamodules.pipes.control.condition import In
from fz_openqa.utils.datastruct import Batch


class ComputeCuiRank(Pipe):
    def __init__(
        self,
        question_cui_key: str = "question.cui",
        document_cui_key: str = "document.cui",
        document_score_key: str = "document.proposal_score",
        output_key: str = "question.document_rank",
        output_match_id_key: str = "question.matched_document_idx",
        method: str = "sum",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.question_cui_key = question_cui_key
        self.document_cui_key = document_cui_key
        self.document_score_key = document_score_key
        self.output_match_id_key = output_match_id_key
        self.method = method
        self.output_key = output_key

    @staticmethod
    def _lower(x: Any) -> Any:
        if isinstance(x, str):
            return x.lower()
        elif isinstance(x, list):
            return [ComputeCuiRank._lower(y) for y in x]
        else:
            raise ValueError(f"Unsupported type {type(x)}")

    @staticmethod
    def _rank(
        q_cuis: List[str], d_cuis: List[str], d_scores: List[float], method: str = "sum"
    ) -> Tuple[int, List[int]]:
        rank_data = [
            ComputeCuiRank._rank_one_q_cui(q_cui, d_cuis, d_scores, method=method)
            for q_cui in q_cuis
        ]
        rank_data = [(r, ids) for r, ids in rank_data if r is not None]
        if len(rank_data) == 0:
            return -1, [-1]
        else:
            ranks, ids = min(rank_data, key=lambda x: x[0])
            return ranks, ids

    @staticmethod
    def _rank_one_q_cui(
        q_cui: str, d_cuis: List[str], d_scores: List[float], method: str = "sum"
    ) -> Tuple[Optional[int], List[int]]:
        """Compute the rank of a query cui in a list of document cuis and
        return the rank along the index of the matched documents."""
        if q_cui not in d_cuis:
            return None, []
        init_value = 0.0 if method == "sum" else -float("inf")
        cui_scores = defaultdict(lambda: init_value)
        cui_ids = defaultdict(list)
        for i, (d_cui, d_score) in enumerate(zip(d_cuis, d_scores)):
            cui_ids[d_cui] += [i]
            if method == "sum":
                if d_score < -1e3:
                    # NB: This is a hack to deal with -float("inf") values.
                    logger.warning(f"Score {d_score} is low. Skipping.")
                else:
                    cui_scores[d_cui] += d_score
            else:
                cui_scores[d_cui] = max(cui_scores[d_cui], d_score)

        # sort document CUIs by score
        ranked_cuis = sorted(cui_scores.items(), key=lambda x: x[1], reverse=True)
        ranked_cuis = [x[0] for x in ranked_cuis]

        # return the rank of the question cui
        try:
            return ranked_cuis.index(q_cui), cui_ids.get(q_cui, [])
        except ValueError:
            return None, []

    def _call_batch(self, batch: Batch, idx: Optional[List[int]] = None, **kwargs) -> Batch:
        """Compute the rank of the query CUIs in the retrieved documents and
        return the ids of the corresponding documents.

        Notes
        -----
        Documents ids correspond to the index of the document in the batch, not the index
        of the document in the dataset."""
        q_batch_cuis = self._lower(batch[self.question_cui_key])
        d_batch_cuis = self._lower(batch[self.document_cui_key])
        d_batch_scores = batch[self.document_score_key]

        # compute ranks and fetch the corresponding document ids
        document_ranks = []
        document_match_ids = []
        for q_cuis, d_cuis, d_scores in zip_longest(
            q_batch_cuis,
            d_batch_cuis,
            d_batch_scores,
        ):
            # get the rank of the question CUIs in the retrieved documents
            rank, matched_ids = self._rank(q_cuis, d_cuis, d_scores, method=self.method)

            document_ranks += [rank]
            document_match_ids += [matched_ids]

        # format output and return
        output = {self.output_key: document_ranks, self.output_match_id_key: document_match_ids}

        return output


class FetchCuiAndRank(Sequential):
    def __init__(
        self,
        corpus_dataset: datasets.Dataset,
        *,
        question_cui_key: str = "question.cui",
        document_cui_key: str = "document.cui",
        document_score_key: str = "document.proposal_score",
        document_id_key: str = "document.row_idx",
        output_key: str = "question.document_rank",
        output_match_id_key: str = "question.matched_document_idx",
        method: str = "sum",
        fetch_max_chunk_size: int = 100,
        level: int = 1,
        **kwargs,
    ):
        super().__init__(
            ApplyAsFlatten(
                FetchDocuments(
                    corpus_dataset=corpus_dataset,
                    keys=[document_cui_key],
                    max_chunk_size=fetch_max_chunk_size,
                ),
                input_filter=In([document_id_key]),
                update=True,
                level=level,
            ),
            ComputeCuiRank(
                question_cui_key=question_cui_key,
                document_cui_key=document_cui_key,
                document_score_key=document_score_key,
                output_key=output_key,
                output_match_id_key=output_match_id_key,
                method=method,
            ),
            **kwargs,
        )
