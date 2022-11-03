import rich
from datasets import Dataset
from warp_pipes import Batch
from warp_pipes import Pipe
from warp_pipes.support.datasets_utils import get_column_names

from .flatten_multiple_choice import FlattenMultipleChoice


class SelectCorrectOption(Pipe):
    def __init__(self, *, key="answer.target", **kwargs):
        super().__init__(**kwargs)
        self.key = key

    def _call_batch(self, batch: Batch, **kwargs) -> Batch:
        options = batch.pop(self.key)
        output = {}
        for key, values in batch.items():
            output[key] = [v[o] for v, o in zip(values, options)]

        return output


class CollapseMultipleChoice(FlattenMultipleChoice):
    def __init__(
        self,
        filter_unmatched: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, filter_unmatched=filter_unmatched, **kwargs)

    def _flatten_dataset(self, dataset: Dataset, map_kwargs):
        # drop columns
        cols_to_remove = ["dataset.idx", "question.idx", "question.metamap", "question.row_idx"]
        cols_to_remove = [c for c in cols_to_remove if c in get_column_names(dataset)]
        dataset = dataset.remove_columns(cols_to_remove)
        # flatten the dataset
        dataset = dataset.map(
            SelectCorrectOption(key="answer.target"),
            desc="Collapse Multiple Choice dataset",
            batched=True,
            **map_kwargs,
        )
        return dataset
