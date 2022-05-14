from datasets import Dataset

from .flatten_multiple_choice import FlattenMultipleChoice
from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.utils.datastruct import Batch


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
        dataset = dataset.remove_columns(["question.idx", "question.metamap", "question.row_idx"])
        # flatten the dataset
        dataset = dataset.map(
            SelectCorrectOption(key="answer.target"),
            desc="Collapse Multiple Choice dataset",
            batched=True,
            **map_kwargs,
        )
        return dataset
