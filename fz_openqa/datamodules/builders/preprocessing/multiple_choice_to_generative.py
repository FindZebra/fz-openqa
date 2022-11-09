from pathlib import Path
from typing import List
from typing import Optional

from datasets import Dataset
from warp_pipes import Batch
from warp_pipes import Eg
from warp_pipes import HasPrefix
from warp_pipes import Pipe

from .base import DatasetPreprocessing


class SelectCorrectOptionPipe(Pipe):
    @classmethod
    def instantiate_test(cls, *, cache_dir: Path, **kwargs) -> Optional["Pipe"]:
        pass

    def __init__(self, *, field="answer", key="target", **kwargs):
        super().__init__(**kwargs)
        self.field = field
        self.key = key

    def _call_batch(self, batch: Batch, **kwargs) -> Batch:
        options = batch.pop(f"{self.field}.{self.key}")
        output = {}
        for key, values in batch.items():
            output[key] = [v[o] for v, o in zip(values, options)]

        return output

    def _call_egs(self, examples: List[Eg], idx: Optional[List[int]] = None, **kwargs) -> Batch:
        raise NotImplementedError("Not implemented for examples")


class MultipleChoiceToGenerative(DatasetPreprocessing):
    def __init__(
        self,
        field: str = "answer",
        key: str = "target",
        only_train_set: bool = False,
        **kwargs,
    ):
        super(MultipleChoiceToGenerative, self).__init__(only_train_set=only_train_set, **kwargs)
        self.field = field
        self.key = key

        self.answer_target_key = f"{self.field}.{self.key}"

        self.select_pipe = SelectCorrectOptionPipe(
            field=self.field,
            key=self.key,
            input_filter=HasPrefix(self.field),
        )

    def preprocess(self, dataset: Dataset, **kwargs) -> Dataset:
        if self.answer_target_key not in dataset.column_names:
            raise ValueError(
                f"Missing column: {self.answer_target_key}. Found: {dataset.column_names}"
            )

        output = self.select_pipe(dataset, **kwargs, remove_columns=[self.answer_target_key])

        return output
