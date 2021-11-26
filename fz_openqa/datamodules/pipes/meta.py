import abc
from collections import OrderedDict
from copy import copy
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import T
from typing import Tuple
from typing import Union

from ...utils.pretty import repr_batch
from .base import Pipe
from .control.condition import HasPrefix
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.functional import check_equal_arrays


class PipeProcessError(Exception):
    """Base class for other exceptions"""

    def __init__(self, meta_pipe: Pipe, pipe: Pipe, batch: Batch, **kwargs):

        try:
            batch_repr = repr_batch(batch)
        except Exception:
            batch_repr = type(batch)

        keys = _infer_keys(batch)
        msg = (
            f"Exception thrown by pipe: {type(pipe)} in meta pipe {type(meta_pipe)} with "
            f"batch of type {type(batch)} with keys={keys} "
            f"and kwargs={kwargs}. Batch=\n{batch_repr}"
        )
        super().__init__(msg)


def _call_pipe_and_handle_exception(
    pipe: Pipe, batch: Batch, meta_pipe: Pipe = None, **kwargs
) -> Batch:
    try:
        return pipe(batch, **kwargs)
    except PipeProcessError as e:
        raise e
    except Exception as e:
        raise PipeProcessError(meta_pipe, pipe, batch, **kwargs) from e


def _infer_keys(batch):
    if isinstance(batch, dict):
        keys = list(batch.keys())
    elif isinstance(batch, list):
        eg = batch[0]
        keys = _infer_keys(eg)
    else:
        keys = [f"<couldn't infer keys, leaf type={type(batch)}>"]
    return keys


class MetaPipe(Pipe):
    """A class that executes other pipes (Sequential, Parallel)"""

    __metaclass__ = abc.ABCMeta

    def _call_batch(self, batch: T, **kwargs) -> T:
        return self._call_all_types(batch, **kwargs)

    def _call_egs(self, batch: T, **kwargs) -> T:
        return self._call_all_types(batch, **kwargs)

    @abc.abstractmethod
    def _call_all_types(self, batch: Union[List[Batch], Batch], **kwargs) -> Batch:
        """
        _call_batch and _call_egs depends on the pipes stored as attributes.
        Therefore a single method can be implemented.
        Only this method should must implemented by the subclasses.
        """
        raise NotImplementedError


class Sequential(MetaPipe):
    """Execute a sequence of pipes."""

    def __init__(self, *pipes: Optional[Union[Callable, Pipe]], **kwargs):
        super(Sequential, self).__init__(**kwargs)
        self.pipes = [pipe for pipe in pipes if pipe is not None]

    def _call_all_types(self, batch: Union[List[Batch], Batch], **kwargs) -> Batch:
        """Call the pipes sequentially."""
        for pipe in self.pipes:
            batch = _call_pipe_and_handle_exception(pipe, batch, **kwargs, meta_pipe=self)

        return batch

    def output_keys(self, input_keys: List[str]) -> List[str]:
        for p in self.pipes:
            input_keys = p.output_keys(input_keys)
        return input_keys


class Parallel(Sequential):
    """Execute pipes in parallel and merge the outputs"""

    def _call_all_types(self, batch: Union[List[Batch], Batch], **kwargs) -> Batch:
        """Call the pipes sequentially."""

        outputs = {}
        for pipe in self.pipes:
            pipe_out = _call_pipe_and_handle_exception(pipe, copy(batch), **kwargs, meta_pipe=self)

            # check conflict between pipes
            o_keys = set(outputs.keys())
            pipe_o_keys = set(pipe_out.keys())
            intersection = o_keys.intersection(pipe_o_keys)
            for key in intersection:
                msg = (
                    f"There is a conflict between pipes on key={key}\n"
                    f"\n{repr_batch(outputs, 'outputs', rich=False)}"
                    f"\n{repr_batch(pipe_out, 'pipe output', rich=False)}"
                )
                assert check_equal_arrays(outputs[key], pipe_out[key]), msg

            # update output
            outputs.update(**pipe_out)

        return outputs

    def output_keys(self, input_keys: List[str]) -> List[str]:
        output_keys = []
        for p in self.pipes:
            p_keys = p.output_keys(input_keys)
            assert all(k not in output_keys for k in p_keys), "There is a conflict between pipes."
            output_keys += p_keys
        return output_keys


class Gate(MetaPipe):
    """Execute the pipe if the condition is valid, else execute alt."""

    def __init__(
        self,
        condition: Union[bool, Callable],
        pipe: Optional[Pipe],
        alt: Optional[Pipe] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.condition = condition
        if isinstance(condition, bool) and condition is False:
            self.pipe = None
            self.alt = alt
        elif isinstance(condition, bool) and condition is True:
            self.pipe = pipe
            self.alt = None
        else:
            self.pipe = pipe
            self.alt = alt

    def output_keys(self, input_keys: List[str]) -> List[str]:
        batch = {k: None for k in input_keys}
        switched_on = self.is_switched_on(batch)
        if switched_on:
            output = self.pipe.output_keys(input_keys)
        elif self.alt is not None:
            output = self.alt.output_keys(input_keys)
        else:
            output = []

        if self.update:
            output = list(set(input_keys + output))

        return output

    def _call_all_types(self, batch: Union[List[Batch], Batch], **kwargs) -> Batch:

        switched_on = self.is_switched_on(batch)

        if switched_on:
            if self.pipe is not None:
                return _call_pipe_and_handle_exception(self.pipe, batch, **kwargs, meta_pipe=self)
            else:
                return {}
        else:
            if self.alt is not None:
                return _call_pipe_and_handle_exception(self.alt, batch, **kwargs, meta_pipe=self)
            else:
                return {}

    def is_switched_on(self, batch):
        if isinstance(self.condition, (bool, int)):
            switched_on = self.condition
        else:
            switched_on = self.condition(batch)
        return switched_on


class BlockSequential(MetaPipe):
    """A sequence of Pipes organized into blocks"""

    def __init__(self, blocks: List[Tuple[str, Pipe]], **kwargs):
        super(BlockSequential, self).__init__(**kwargs)
        blocks = [(k, b) for k, b in blocks if b is not None]
        self.blocks: OrderedDict[str, Pipe] = OrderedDict(blocks)

    def _call_all_types(self, batch: Union[List[Batch], Batch], **kwargs) -> Batch:
        """Call the pipes sequentially."""
        for block in self.blocks.values():
            batch = _call_pipe_and_handle_exception(block, batch, **kwargs, meta_pipe=self)

        return batch

    def output_keys(self, input_keys: List[str]) -> List[str]:
        for _, p in self.blocks.items():
            input_keys = p.output_keys(input_keys)
        return input_keys


class ParallelbyField(Parallel):
    """Run a pipe for each field"""

    def __init__(self, pipes: Dict[str, Pipe], **kwargs):
        super(ParallelbyField, self).__init__(**kwargs)
        self.pipes = {
            field: Sequential(pipe, input_filter=HasPrefix(f"{field}."))
            for field, pipe in pipes.items()
            if pipe is not None
        }
