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
from fz_openqa.utils.functional import check_equal_arrays


def safe_dill_inspect(p):
    if isinstance(p, Pipe):
        return p.dill_inspect()
    else:
        return dill.pickles(p)


class Sequential(Pipe):
    """A sequence of Pipes."""

    def __init__(self, *pipes: Optional[Union[Callable, Pipe]], id: Optional[str] = None):
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

    def dill_inspect(self, reduce: bool = False) -> Union[Dict[str, Any], bool]:
        diagnostic = {self.get_pipe_id(p): safe_dill_inspect(p) for p in self.pipes}
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

        outputs = {}
        for pipe in self.pipes:
            pipe_out = pipe(batch, **kwargs)

            # check conflict between pipes
            o_keys = set(outputs.keys())
            pipe_o_keys = set(pipe_out.keys())
            intersection = o_keys.intersection(pipe_o_keys)
            for key in intersection:
                msg = (
                    f"There is a conflict between pipes on key={key}\n"
                    f"- outputs: {outputs[key]}\n"
                    f"- pipe output: {pipe_out[key]}"
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
    """Execute the pipe if the condition is valid, else execute alt."""

    def __init__(
        self,
        condition: Union[bool, Callable],
        pipe: Optional[Pipe],
        alt: Optional[Pipe] = None,
        id: Optional[str] = None,
        update: bool = False,
    ):
        super().__init__(id=id)

        self.update = update
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

    @property
    def id(self):
        return str(type(self.pipe).__name__)

    def __call__(self, batch: Union[List[Batch], Batch], **kwargs) -> Batch:
        output = self._call(batch, **kwargs)
        if self.update:
            batch.update(output)
            output = batch
        return output

    def _call(self, batch: Union[List[Batch], Batch], **kwargs) -> Batch:

        switched_on = self.is_switched_on(batch)

        if switched_on:
            if self.pipe is not None:
                return self.pipe(batch, **kwargs)
            else:
                return {}
        else:
            if self.alt is not None:
                return self.alt(batch, **kwargs)
            else:
                return {}

    def is_switched_on(self, batch):
        if isinstance(self.condition, (bool, int)):
            switched_on = self.condition
        else:
            switched_on = self.condition(batch)
        return switched_on

    def todict(self) -> Dict:
        d = super().todict()
        d["pipe"] = safe_todict(self.pipe)
        d["alt"] = safe_todict(self.alt)
        return d

    def dill_inspect(self, reduce: bool = False) -> Any:
        return self.pipe.dill_inspect()

    def fingerprint(self) -> Any:
        return safe_fingerprint(self.pipe)

    def __repr__(self):
        data = self.todict()
        data["condition"] = str(data["condition"])
        data["pipe"] = self.pipe.todict()
        data["alt"] = self.alt.todict()
        return json.dumps(data, indent=4)


class BlockSequential(Pipe):
    """A sequence of Pipes organized into blocks"""

    def __init__(self, blocks: List[Tuple[str, Pipe]], id: Optional[str] = None):
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

    def dill_inspect(self, reduce: bool = False) -> Union[Dict[str, Any], bool]:
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
