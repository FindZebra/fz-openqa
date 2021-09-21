from typing import List
from typing import Optional
from typing import Union

from fz_openqa.datamodules.pipes.base import Pipe
from fz_openqa.datamodules.pipes.collate import Collate
from fz_openqa.utils.datastruct import Batch


class Sequential(Pipe):
    """A sequence of Pipes."""

    def __init__(self, *pipes: Optional[Union[Collate, Pipe]]):
        self.pipes = [pipe for pipe in pipes if pipe is not None]

    def __call__(self, batch: Union[List[Batch], Batch], **kwargs) -> Batch:
        """Call the pipes sequentially."""
        for pipe in self.pipes:
            batch = pipe(batch, **kwargs)

        return batch


class Parallel(Pipe):
    """Execute pipes in parallel and merge."""

    def __init__(self, *pipes: Optional[Union[Collate, Pipe]]):
        self.pipes = [pipe for pipe in pipes if pipe is not None]

    def __call__(self, batch: Union[List[Batch], Batch], **kwargs) -> Batch:
        """Call the pipes sequentially."""

        output = {}
        for pipe in self.pipes:
            pipe_out = pipe(batch, **kwargs)
            assert all(
                k not in output.keys() for k in pipe_out.keys()
            ), "There is a conflict between pipes."
            output.update(**pipe_out)

        return output
