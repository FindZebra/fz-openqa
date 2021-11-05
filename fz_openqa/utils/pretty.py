import shutil
from typing import List
from typing import Union

import numpy as np
import rich
from torch import Tensor
from transformers import PreTrainedTokenizerFast

from fz_openqa.utils.datastruct import Batch


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
    u = ""
    if header is not None:
        u += get_separator() + "\n"
        u += f"=== {header} ===\n"

    u += get_separator() + "\n"
    u += f"Batch <{type(batch).__name__}>:"
    for k in sorted(batch.keys()):
        v = batch[k]
        if isinstance(v, Tensor):
            u += f"\n   - {k}: {v.shape} <{v.dtype}> ({v.device})"
        elif isinstance(v, np.ndarray):
            u += f"\n   - {k}: {v.shape} <{v.dtype}>"
        elif isinstance(v, list):
            if isinstance(v[0], str):
                lens = [len(vv) for vv in v]
                u += (
                    f"\n   - {k}: {len(v)} items with {min(lens)} "
                    f"to {max(lens)} characters <list<text>>"
                )
            elif isinstance(v[0], list):
                dtype = type(v[0][0]).__name__ if len(v[0]) else "<empty>"
                lengths = list(set([len(vv) for vv in v]))
                if len(lengths) == 1:
                    w = f"shape = [{len(v)}, {lengths[0]}]"
                else:
                    w = f"{len(v)} items, each with {min(lengths)} to {max(lengths)} elements"
                u += f"\n   - {k}: {w} <list<list<{dtype}>>>"
            else:
                dtype = type(v[0]).__name__
                u += f"\n   - {k}: {len(v)} items <list<{dtype}>>"
        else:
            u += f"\n   - {k}: {v} {type(v)}"

    u += "\n" + get_separator()
    rich.print(u)
