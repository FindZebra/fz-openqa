from functools import partial
from typing import Optional

from datasets import DatasetDict
from loguru import logger

from ...pipes import Flatten
from ...pipes import SupervisedDatasetFilter
from ...utils.datastruct import OpenQaConfig
from ...utils.transformations import set_index_column
from ...utils.typing import HfDataset
from .base import OpenQaTransform


class FlattenMultipleChoice(OpenQaTransform):
    def __init__(
        self,
        filter_unmatched: bool = True,
        match_score_key: str = "document.match_score",
        max_positives: int = None,
    ):
        if filter_unmatched:
            self.dataset_filter = SupervisedDatasetFilter(
                key=match_score_key, max_positives=max_positives
            )
        else:
            self.dataset_filter = None

    def _transform_config(self, openqa_config: OpenQaConfig, **kwargs) -> OpenQaConfig:
        # nesting will be reduced by one level
        assert openqa_config.question_nesting_level == 1
        assert openqa_config.document_nesting_level == 2
        openqa_config.question_nesting_level -= 1
        openqa_config.document_nesting_level -= 1
        return openqa_config

    def _transform_dataset(
        self, dataset: HfDataset, openqa_config: Optional[OpenQaConfig], **map_kwargs
    ) -> HfDataset:
        """
        Flatten: Reduce nesting by one level :
        [n_questions, n_options, *] -> [n_questions * n_options, *]

        (Optional) Filter: Remove unmatched documents
        """

        lengths = {split: len(dset) for split, dset in dataset.items()}
        dataset = self._flatten_dataset(dataset, map_kwargs)
        self._log_diff_length("Flatten", dataset, lengths)

        if self.dataset_filter is not None:
            lengths = {split: len(dset) for split, dset in dataset.items()}
            dataset = self._filter_dataset(dataset, map_kwargs)
            self._log_diff_length("Filter", dataset, lengths)

        # reset question.idx column
        dataset = set_index_column(dataset, key="question.row_idx")

        return dataset

    def _flatten_dataset(self, dataset, map_kwargs):
        # drop columns
        dataset = dataset.remove_columns(
            ["answer.target", "question.idx", "question.metamap", "question.row_idx"]
        )
        # flatten the dataset
        dataset = dataset.map(
            Flatten(level=1), desc="Flatten Multiple Choice dataset", batched=True, **map_kwargs
        )
        return dataset

    def _filter_dataset(self, dataset, map_kwargs):
        # filter out unmatched documents
        dataset = DatasetDict(
            {
                split: dset.filter(
                    partial(self.dataset_filter, split=split), batched=True, **map_kwargs
                )
                for split, dset in dataset.items()
            }
        )

        return dataset

    def _log_diff_length(self, desc, dataset, prev_lengths):
        for split, prev_length in prev_lengths.items():
            logger.info(
                f"{desc} ({split}): dataset length: {prev_length} "
                f"-> {len(dataset[split])} "
                f"({len(dataset[split]) / prev_length:.2%})"
            )
