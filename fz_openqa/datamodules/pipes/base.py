import logging
from abc import abstractmethod
from copy import copy
from functools import singledispatchmethod
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import T

from datasets import Dataset
from datasets import DatasetDict

from fz_openqa.datamodules.component import Component
from fz_openqa.datamodules.pipes.control.condition import Condition
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.datastruct import Eg
from fz_openqa.utils.functional import get_batch_eg

logger = logging.getLogger(__name__)


class Pipe(Component):
    """
    A pipe is a small unit of computation that ingests,
    modify and returns a batch of data.

    ----------
    Attributes
    id
       An identifier for the pipe.
    input_filter
        Condition used to filter keys in the input data.
    update
        If set to True, output the input batch with the output batch.
    requires_keys
       A list of keys that the pipe requires to be present in the data.
    """

    # __metaclass__ = ABCMeta
    id: Optional[str] = None
    input_filter: Optional[Condition] = None
    requires_keys: Optional[List[str]] = None
    _allows_update: bool = True
    _allows_input_filter: bool = True
    _backend: Optional[str] = None

    def __init__(
        self,
        *,
        id: Optional[str] = None,
        input_filter: Optional[Condition] = None,
        update: bool = False,
    ):
        """
        Parameters
        ----------
        id
           An identifier for the pipe.
        input_filter
            a condition used to filter keys in the input data
            (keys that do not satisfy the condition are removed)
        update
            If set to True, output the input batch updated with the output batch.
        """
        super().__init__(id=id)
        if not self._allows_update and update:
            raise AttributeError(f"{type(self).__name__} does not allow using update=True")

        if not self._allows_input_filter and input_filter is not None:
            raise AttributeError(f"{type(self).__name__} does not allow using input_filter")

        if input_filter is not None:
            self.input_filter = input_filter
        self.update = update

    def output_keys(self, input_keys: List[str]) -> List[str]:
        """
        Return the list of keys that the pipe is expected to return.
        Parameters
        ----------
        input_keys
           The list of keys that the pipe expects as input.

        Returns
        -------
        List[str]
           The list of keys that the pipe will output
        """
        output_keys = copy(input_keys)
        if self.input_filter is not None:
            output_keys = list({k: None for k in output_keys if self.input_filter(k)}.keys())

        if self.update:
            output_keys = input_keys + output_keys

        return output_keys

    @staticmethod
    def get_eg(batch: Batch, idx: int, filter_op: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Extract example `idx` from a batch, potentially filter keys.

        Parameters
        ----------
        batch
           Input batch
        idx
           Index of the example to extract
        filter_op
           A function that used to filter the keys

        Returns
        -------
        Dict[str, Any]
           The example of rank `idx`
        """
        return get_batch_eg(batch=batch, idx=idx, filter_op=filter_op)

    @singledispatchmethod
    def __call__(self, data: T, **kwargs) -> T:
        """
        Apply the pipe to a data. Potentially filter the keys using the input_filter.
        This method is dispatched on the type of the input data.

        Parameters
        ----------
        data
            The input data
        idx
            indexes of the batch examples
        kwargs
            additional arguments

        Returns
        -------
        Batch
            The output data
        """

        raise TypeError(f"{type(self).__name__} does not support {type(data).__name__}.")

    @__call__.register(dict)
    def _(self, batch: Batch, idx: Optional[List[int]] = None, **kwargs) -> Batch:
        """
        Apply the pipe to a batch of data. Potentially filter the keys using the input_filter.
        The output of `_call_batch()` is used to update the input batch (before filtering)
        if update=True, else the raw output is returned.

        Parameters
        ----------
        batch
            batch to apply the pipe to
        idx
            indexes of the batch examples
        kwargs
            additional arguments

        Returns
        -------
        Batch
            The output batch
        """

        # filter some input keys
        _batch = self._filter_keys(batch)

        # process the batch
        output = self._call_batch(_batch, idx=idx, **kwargs)

        # update the input batch with the output if update is set to True
        if self.update:
            batch.update(output)
            output = batch

        return output

    @__call__.register(list)
    def _(self, examples: List[Eg], idx: Optional[List[int]] = None, **kwargs) -> Batch:
        """
        Apply the pipe to a list of examples. Typically to concatenate examples.

        Parameters
        ----------
        examples
            batch of examples to apply the pipe to
        idx
            indexes of the examples
        kwargs
            additional arguments

        Returns
        -------
        Batch
            The output batch
        """

        if not all(isinstance(eg, dict) for eg in examples):
            raise TypeError(
                f"Error in pipe {type(self)}, examples must be a list of dicts, "
                f"got {type(examples[0])}"
            )

        if self.update is True:
            raise AttributeError("Pipe.update is set to True, cannot update a list of examples")

        # filter some input keys
        _egs = list(map(self._filter_keys, examples))

        # process the batch
        return self._call_egs(_egs, idx=idx, **kwargs)

    @__call__.register(Dataset)
    def _(
        self,
        dataset: Dataset,
        *,
        num_proc: int = 4,
        desc: Optional[str] = None,
        batch_size: Optional[int] = None,
        writer_batch_size: Optional[int] = None,
        **kwargs,
    ) -> Dataset:
        """
        Apply the Pipe to a `Dataset`

        Parameters
        ----------
        dataset
            A Huggingface Dataset object
        num_proc
            Number of workers
        desc
            Description for the progress bar
        batch_size
            Batch size for each worker
        writer_batch_size
            Batch size for the pyarrow writer
        kwargs
            Additional attributes passed to the pipe
        Returns
        -------
        Dataset
            Processed dataset
        """
        return self._call_dataset(
            dataset,
            num_proc=num_proc,
            desc=desc,
            batch_size=batch_size,
            writer_batch_size=writer_batch_size,
            **kwargs,
        )

    @__call__.register(DatasetDict)
    def _(self, dataset: DatasetDict, **kwargs) -> DatasetDict:
        """
        Apply the Pipe to a `DatasetDict`

        Parameters
        ----------
        dataset
            A Huggingface DatasetDict object
        kwargs
            Additional attributes passed to `_call_dataset`
        Returns
        -------
        Dataset
            Processed dataset
        """
        new_datasets = {split: self._call_dataset(d, **kwargs) for split, d in dataset.items()}
        return DatasetDict(new_datasets)

    @abstractmethod
    def _call_batch(self, batch: Batch, idx: Optional[List[int]] = None, **kwargs) -> Batch:
        """
        Main operation applied to the batch.

        Parameters
        ----------
        batch
            batch to apply the pipe to
        idx
            indexes of the batch examples
        kwargs
            additional arguments

        Returns
        -------
        Batch
            The output batch
        """
        raise NotImplementedError(f"_call_batch is not implemented for {type(self)}")

    @abstractmethod
    def _call_egs(self, examples: List[Eg], idx: Optional[List[int]] = None, **kwargs) -> Batch:
        """
        Main Operation applied to a list of examples (Egs). Typically to concatenate examples.

        Parameters
        ----------
        examples
            List of examples
        idx
            indexes of the examples
        kwargs
            additional arguments

        Returns
        -------
        Batch
            The output batch
        """
        raise NotImplementedError(f"_call_egs is not implemented for {type(self)}")

    def _filter_keys(self, batch: Batch) -> Batch:
        """
        Filter the batch using the input_filter.

        Parameters
        ----------
        batch
            batch to filter

        Returns
        -------
        Batch
            Filtered batch
        """
        if self.input_filter is None:
            return batch

        return {k: v for k, v in batch.items() if self.input_filter(k)}

    def _call_dataset(
        self,
        dataset: Dataset,
        *,
        num_proc: int = 4,
        desc: Optional[str] = None,
        batch_size: Optional[int] = None,
        writer_batch_size: Optional[int] = None,
        **kwargs,
    ) -> Dataset:
        """
        Apply the Pipe to a `Dataset`

        Parameters
        ----------
        dataset
            A Huggingface Dataset object
        num_proc
            Number of workers
        desc
            Description for the progress bar
        batch_size
            Batch size for each worker
        writer_batch_size
            Batch size for the pyarrow writer
        kwargs
            Additional attributes passed to the pipe
        Returns
        -------
        Dataset
            Processed dataset
        """
        for key in ["batched", "with_indices"]:
            if key in kwargs.keys():
                raise ValueError(f"{key} cannot be set, it is always set as True.")

        return dataset.map(
            self,
            num_proc=num_proc,
            desc=desc,
            batch_size=batch_size,
            batched=True,
            with_indices=True,
            writer_batch_size=writer_batch_size,
            **kwargs,
        )
