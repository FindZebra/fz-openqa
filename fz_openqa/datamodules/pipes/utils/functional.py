from numbers import Number
from typing import Any
from typing import Dict
from typing import Union

from fz_openqa.datamodules.pipes import Pipe


def safe_todict(x):
    """return a dictionary representation of `x`, even if `x` is not a `Pipe`."""
    if isinstance(x, Pipe):
        return x.todict()
    else:
        try:
            return dict(x)
        except Exception:
            return {"__obj_str__": str(x)}


def safe_fingerprint(x):
    """Return the fingerprint of `x`, even if `x` is not a `Pipe`."""
    if isinstance(x, Pipe):
        return x.fingerprint()
    else:
        return Pipe._fingerprint(x)


def reduce_dict_values(x: Union[bool, Dict[str, Any]], op=all) -> bool:
    """Reduce a nested dictionary structure with boolean values
    into a single boolean output."""
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
        assert isinstance(x, (bool, Number))
        return x
