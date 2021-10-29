from typing import Callable
from typing import Union

from datasets import Dataset
from torch.nn import Module

from fz_openqa.datamodules.index.base import Index
from fz_openqa.datamodules.index.base import SearchResult
from fz_openqa.datamodules.pipes import FilterKeys
from fz_openqa.datamodules.pipes import Forward
from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.datamodules.pipes import Sequential
from fz_openqa.datamodules.pipes import ToNumpy
from fz_openqa.utils.datastruct import Batch


class FaissIndex(Index):
    """Dense indexing using faiss"""

    vectors_column_name = "__vectors__"
    dataset: Dataset = None
    model: Callable = None

    def __init__(self, dataset: Dataset, *, batch_size: int = 32, **kwargs):
        self.batch_size = batch_size
        super(FaissIndex, self).__init__(dataset=dataset, **kwargs)

    def build(
        self,
        dataset: Dataset,
        *,
        model: Union[Callable, Module] = None,
        **kwargs,
    ):
        """Index a dataset."""
        assert model is not None
        self.model = model
        self.dataset = dataset.map(
            self.get_batch_processing_pipe(model),
            batched=True,
            batch_size=self.batch_size,
            num_proc=1,
            desc="Computing corpus vectors",
            remove_columns=[x for x in dataset.column_names if x != "idx"],
            # remove this if scalability issues
            keep_in_memory=True,
        )

        # add the dense index
        self.dataset.add_faiss_index(
            column=self.vectors_column_name, device=None
        )

        # set status
        self.is_indexed = True

    def search(
        self,
        query: Batch,
        *,
        k: int = 1,
        model: Union[Callable, Module] = None,
        **kwargs,
    ) -> SearchResult:
        """Search the index using the `query` and
        return the index of the results within the original dataset."""
        assert self.dataset is not None
        model = model or self.model
        assert model is not None

        query = self.get_batch_processing_pipe(model=model)(query)

        results = self.dataset.get_nearest_examples_batch(
            self.vectors_column_name, query[self.vectors_column_name], k=k
        )

        return SearchResult(
            score=results.total_scores,
            index=[r["idx"] for r in results.total_examples],
        )

    def get_batch_processing_pipe(
        self, model: Union[Callable, Module]
    ) -> Pipe:
        """returns a pipe that allows processing a batch of data using the model."""
        return Sequential(
            Forward(model=model, output_key=self.vectors_column_name),
            FilterKeys(lambda key: key == self.vectors_column_name),
            ToNumpy(),
        )
