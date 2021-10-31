import json
from collections import OrderedDict
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import dill

from fz_openqa.datamodules.pipes.base import Pipe
from fz_openqa.datamodules.pipes.utils import reduce_dict_values
from fz_openqa.datamodules.pipes.utils import safe_fingerprint
from fz_openqa.datamodules.pipes.utils import safe_todict
from fz_openqa.utils.datastruct import Batch


def safe_dill_inspect(p):
    if isinstance(p, Pipe):
        return p.dill_inspect()
    else:
        return dill.pickles(p)


class Sequential(Pipe):
    """A sequence of Pipes."""

    def __init__(
        self, *pipes: Optional[Union[Callable, Pipe]], id: Optional[str] = None
    ):
        super(Sequential, self).__init__(id=id)
        self.pipes = [pipe for pipe in pipes if pipe is not None]

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
            self.get_pipe_id(p): safe_dill_inspect(p) for p in self.pipes
        }
        if reduce:
            return reduce_dict_values(diagnostic)
        else:
            return diagnostic

    def fingerprint(self) -> Dict[str, Any]:
        return {self.get_pipe_id(p): safe_fingerprint(p) for p in self.pipes}

    def output_keys(self, input_keys: List[str]) -> List[str]:
        for p in self.pipes:
            input_keys = p.output_keys(input_keys)
        return input_keys

    def __repr__(self):
        data = self.todict()
        try:
            return json.dumps(data, indent=4)
        except Exception:
            return str(data)


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

    def output_keys(self, input_keys: List[str]) -> List[str]:
        output_keys = []
        for p in self.pipes:
            p_keys = p.output_keys(input_keys)
            assert all(
                k not in output_keys for k in p_keys
            ), "There is a conflict between pipes."
            output_keys += p_keys
        return output_keys


class UpdateWith(Pipe):
    def __init__(self, pipe: Pipe, **kwargs):
        super(UpdateWith, self).__init__(**kwargs)
        self.pipe = pipe

    def __call__(self, batch: Union[List[Batch], Batch], **kwargs) -> Batch:
        """Call the pipes sequentially."""

        batch.update(self.pipe(batch, **kwargs))
        # output = self.pipe(batch, **kwargs)
        # output.update(**{k: v for k, v in batch.items() if k not in output})
        return batch

    def fingerprint(self) -> Dict[str, Any]:
        return {
            "__self__": self._fingerprint(self),
            "pipe": self.pipe.fingerprint(),
        }

    def dill_inspect(self, reduce: bool = False) -> bool:
        return self.pipe.dill_inspect()

    def output_keys(self, input_keys: List[str]) -> List[str]:
        return self.pipe.output_keys(input_keys)

    def __repr__(self):
        data = self.todict()
        data["pipe"] = self.pipe.todict()
        return json.dumps(data, indent=4)


class Gate(Pipe):
    """Execute the pipe if the condition is valid, else return {}"""

    def __init__(
        self,
        condition: Union[bool, Callable],
        pipe: Optional[Pipe],
        id: Optional[str] = None,
    ):
        super().__init__(id=id)

        self.condition = condition
        self.pipe = pipe

    def output_keys(self, input_keys: List[str]) -> List[str]:
        if self.condition({k: None for k in input_keys}):
            return self.pipe.output_keys(input_keys)
        else:
            return []

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
            return self.pipe(batch, **kwargs)
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

    def __repr__(self):
        data = self.todict()
        data["condition"] = str(data["condition"])
        data["pipe"] = self.pipe.todict()
        return json.dumps(data, indent=4)


class BlockSequential(Pipe):
    """A sequence of Pipes organized into blocks"""

    def __init__(
        self, blocks: List[Tuple[str, Pipe]], id: Optional[str] = None
    ):
        super(BlockSequential, self).__init__(id=id)
        blocks = [(k, b) for k, b in blocks if b is not None]
        self.blocks: OrderedDict[str, Pipe] = OrderedDict(blocks)

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

    def __iter__(self):
        for k, b in self.blocks.items():
            yield (k, b)

    def output_keys(self, input_keys: List[str]) -> List[str]:
        for _, p in self.blocks.items():
            input_keys = p.output_keys(input_keys)
        return input_keys

    def __repr__(self):
        data = self.todict()
        return json.dumps(data, indent=4)
