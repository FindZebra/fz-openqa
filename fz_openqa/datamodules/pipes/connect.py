from collections import OrderedDict
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import rich

from fz_openqa.datamodules.pipes.base import Pipe
from fz_openqa.datamodules.pipes.utils import reduce_dict_values
from fz_openqa.datamodules.pipes.utils import safe_fingerprint
from fz_openqa.datamodules.pipes.utils import safe_todict
from fz_openqa.utils.datastruct import Batch


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

    def todict(self) -> Dict:
        """return a dictionary representation of this pipe."""
        d = super().todict()
        d["pipes"] = [safe_todict(p) for p in self.pipes]
        return d

    @staticmethod
    def get_pipe_id(p: Pipe):
        cls = str(type(p).__name__)
        if not isinstance(p, Pipe) or p.id is None:
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
        return {self.get_pipe_id(p): safe_fingerprint(p) for p in self.pipes}

    def as_fingerprintable(self) -> Any:
        pipes = [p.as_fingerprintable() for p in self.pipes]
        return Sequential(*pipes, id=self.id)


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

    def as_fingerprintable(self) -> Any:
        pipes = [p.as_fingerprintable() for p in self.pipes]
        return Parallel(*pipes, id=self.id)


class UpdateWith(Pipe):
    def __init__(self, pipe: Pipe):
        self.pipe = pipe

    def __call__(self, batch: Union[List[Batch], Batch], **kwargs) -> Batch:
        """Call the pipes sequentially."""

        batch.update(self.pipe(batch, **kwargs))

        return batch

    def dill_inspect(self, reduce: bool = False) -> bool:
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

    def as_fingerprintable(self) -> Pipe:
        return Gate(self.condition, self.pipe.as_fingerprintable())

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

    def todict(self) -> Dict:
        d = super().todict()
        d["pipe"] = safe_todict(self.pipe)
        return d

    def dill_inspect(self, reduce: bool = False) -> Any:
        return self.pipe.dill_inspect()

    def fingerprint(self) -> Any:
        return safe_fingerprint(self.pipe)


class BlockSequential(Pipe):
    """A sequence of Pipes organized into blocks"""

    def __init__(
        self, blocks: List[Tuple[str, Pipe]], id: Optional[str] = None
    ):
        self.blocks: OrderedDict[str, Pipe] = OrderedDict(blocks)
        self.id = id

    def __call__(self, batch: Union[List[Batch], Batch], **kwargs) -> Batch:
        """Call the pipes sequentially."""
        for block in self.blocks.values():
            batch = block(batch, **kwargs)

        return batch

    def todict(self) -> Dict:
        """return a dictionary representation of this pipe."""
        d = super().todict()
        d["blocks"] = {k: safe_todict(p) for k, p in self.blocks.items()}
        return d

    @staticmethod
    def get_pipe_id(p: Pipe):
        cls = str(type(p).__name__)
        if not isinstance(p, Pipe) or p.id is None:
            return cls
        else:
            return f"{cls}({p.id})"

    def dill_inspect(
        self, reduce: bool = False
    ) -> Union[Dict[str, Any], bool]:
        diagnostic = {k: p.dill_inspect() for k, p in self.blocks.items()}
        if reduce:
            return reduce_dict_values(diagnostic)
        else:
            return diagnostic

    def fingerprint(self) -> Dict[str, Any]:
        return {k: safe_fingerprint(p) for k, p in self.blocks.items()}

    def as_fingerprintable(self) -> Any:
        blocks = [(k, p.as_fingerprintable()) for k, p in self.blocks.items()]
        return BlockSequential(blocks, id=self.id)

    def __iter__(self):
        for k, b in self.blocks.items():
            yield (k, b)
