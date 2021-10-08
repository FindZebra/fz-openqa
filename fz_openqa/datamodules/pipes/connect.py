from numbers import Number
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from fz_openqa.datamodules.pipes.base import Pipe
from fz_openqa.utils.datastruct import Batch


def reduce_dict_values(x: Union[bool, Dict[str, Any]], op=all) -> bool:
    if isinstance(x, dict):
        outputs = []
        for v in x.values():
            if isinstance(v, dict):
                outputs += [reduce_dict_values(v)]
            else:
                assert isinstance(v, bool)
                outputs += [v]

        return op(outputs)

    else:
        assert isinstance(
            x,
            (
                bool,
                Number,
            ),
        )
        return x


class Sequential(Pipe):
    """A sequence of Pipes."""

    def __init__(
        self, *pipes: Optional[Union[Callable, Pipe]], id: Optional[str] = None
    ):
        self.pipes = [pipe for pipe in pipes if pipe is not None]
        self.id = id

    def __call__(self, batch: Union[List[Batch], Batch], **kwargs) -> Batch:
        """Call the pipes sequentially."""
        for pipe in self.pipes:
            batch = pipe(batch, **kwargs)

        return batch

    @staticmethod
    def get_pipe_id(p: Pipe):
        cls = str(type(p).__name__)
        if p.id is None:
            return cls
        else:
            return f"{cls}({p.id})"

    def dill_inspect(
        self, reduce: bool = False
    ) -> Union[Dict[str, Any], bool]:
        diagnostic = {
            self.get_pipe_id(p): p.dill_inspect() for p in self.pipes
        }
        if reduce:
            return reduce_dict_values(diagnostic)
        else:
            return diagnostic

    def fingerprint(self) -> Dict[str, Any]:
        return {self.get_pipe_id(p): p.fingerprint() for p in self.pipes}


class Parallel(Sequential):
    """Execute pipes in parallel and merge."""

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


class Update(Pipe):
    def __init__(self, pipe: Pipe):
        self.pipe = pipe

    def __call__(self, batch: Union[List[Batch], Batch], **kwargs) -> Batch:
        """Call the pipes sequentially."""

        batch.update(self.pipe(batch, **kwargs))

        return batch

    def dill_inspect(self) -> bool:
        return self.pipe.dill_inspect()


class Gate(Pipe):
    """Execute the pipe if the condition is valid, else return {}"""

    def __init__(
        self,
        condition: Union[bool, Callable],
        pipe: Optional[Pipe],
    ):
        self.condition = condition
        self.pipe = pipe

    @property
    def id(self):
        return str(type(self.pipe).__name__)

    def __call__(self, batch: Union[List[Batch], Batch], **kwargs) -> Batch:
        """Call the pipes sequentially."""

        if isinstance(self.condition, (bool, int)):
            switched_on = self.condition
        else:
            switched_on = self.condition(batch)

        if switched_on and self.pipe is not None:
            return self.pipe(batch)
        else:
            return {}

    def dill_inspect(self) -> Any:
        return self.pipe.dill_inspect()

    def fingerprint(self) -> Any:
        return self.pipe.fingerprint()
