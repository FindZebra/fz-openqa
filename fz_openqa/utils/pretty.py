import logging
import shutil
from collections import defaultdict
from typing import Dict
from typing import List
from typing import Union

import numpy as np
import rich
import torch
from torch import Tensor
from transformers import PreTrainedTokenizerFast

from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.shape import infer_batch_shape
from fz_openqa.utils.shape import infer_shape

logger = logging.getLogger(__name__)


def pretty_decode(
    tokens: Union[Tensor, List[int], np.ndarray],
    *,
    tokenizer: PreTrainedTokenizerFast,
    style: str = "deep_sky_blue3",
    only_text: bool = False,
    **kwargs,
):
    """Pretty print an encoded chunk of text"""
    if style != "":
        style_in = f"[{style}]"
        style_out = f"[/{style}]"
    else:
        style_in = style_out = ""
    n_pad_tokens = list(tokens).count(tokenizer.pad_token_id)
    txt = tokenizer.decode(tokens, **kwargs)
    txt = f"{style_in}`{txt.replace('[PAD]', '').strip()}`{style_out}"
    if only_text:
        return txt
    else:
        return f"length={len(tokens)}, padding={n_pad_tokens}, " f"text: {txt}"


def get_separator(char="\u2500"):
    console_width, _ = shutil.get_terminal_size()
    return console_width * char


def pprint_batch(batch: Batch, header=None):
    exceptions = _pprint_batch(batch, header)
    for key, e in exceptions.items():
        logger.warning(f"Couldn't pretty print key={key}\nException={e}")


def _pprint_batch(batch: Batch, header=None) -> Dict[str, Exception]:
    u = get_separator() + "\n"
    if header is not None:
        u += f"=== {header} ===\n"
        u += get_separator("-") + "\n"

    u += f"Batch (shape={infer_batch_shape(batch)}):\n"

    data = []
    exceptions = {}
    for k in sorted(batch.keys()):
        try:
            shape, leaf_type = infer_shape(batch[k], return_leaf_type=True)
        except Exception as e:
            exceptions[k] = e
        data += [{"key": k, "shape": str(shape), "leaf_type": str(leaf_type)}]

    keys = list(data[0].keys())
    maxs = {k: max([len(d[k]) for d in data]) for k in keys}
    _s = "  "
    _sep = " [white]|[/white] "
    _row_sep = (
        f"{_s}[white]{'_'*maxs['key']}"
        f"{_sep}{'_'*maxs['shape']}"
        f"{_sep}{'_'*maxs['leaf_type']}[/white]\n"
    )
    u += _row_sep
    u += (
        f"{_s}{'key':{maxs['key']}}"
        f"{_sep}{'shape':{maxs['shape']}}"
        f"{_sep}{'leaf_type':{maxs['leaf_type']}}\n"
    )
    u += _row_sep
    for row in data:
        u += (
            f"{_s}{row['key']:{maxs['key']}}"
            f"{_sep}{row['shape']:{maxs['shape']}}"
            f"{_sep}{row['leaf_type']:{maxs['leaf_type']}}\n"
        )

    u += _row_sep
    rich.print(u)
    if len(exceptions):
        rich.print(
            f"couldn't pretty print keys={list(exceptions.keys())}. See log for further details."
        )
    return exceptions
