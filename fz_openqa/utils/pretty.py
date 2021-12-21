import logging
import shutil
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import rich
from datasets import arrow_dataset
from torch import Tensor
from transformers import PreTrainedTokenizerFast

from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.json_struct import flatten_json_struct
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

    if isinstance(tokens, Tensor):
        tokens = tokens.cpu().numpy()
    if isinstance(tokens, np.ndarray):
        tokens = tokens.tolist()

    n_pad_tokens = tokens.count(tokenizer.pad_token_id)
    txt = tokenizer.decode(tokens, **kwargs)
    txt = f"{style_in}`{txt.replace('[PAD]', '').strip()}`{style_out}"
    if only_text:
        return txt
    else:
        return f"length={len(tokens)}, padding={n_pad_tokens}, " f"text: {txt}"


def get_separator(char="\u2500"):
    console_width, _ = shutil.get_terminal_size()
    return console_width * char


def pprint_batch(
    batch: Batch, header=None, report_nans: bool = False, silent: bool = False
) -> None:
    if silent:
        return
    u, exceptions = _repr_batch(batch, header, report_nans=report_nans, rich=True)
    u = get_separator() + "\n" + u
    rich.print(u)
    if len(exceptions):
        rich.print(
            f"couldn't pretty print keys={list(exceptions.keys())}. See log for further details."
        )
    for key, e in exceptions.items():
        logger.warning(f"Couldn't pretty print key={key}\nException={e}")


def repr_batch(batch: Batch, header=None, report_nans: bool = False, rich: bool = False) -> str:
    u, exceptions = _repr_batch(batch, header, report_nans=report_nans, rich=rich)
    return u


def _repr_batch(
    batch: Batch, header=None, rich: bool = False, report_nans: bool = False
) -> Tuple[str, Dict[str, Exception]]:
    if not isinstance(
        batch,
        (
            dict,
            arrow_dataset.Batch,
        ),
    ):
        return f"Batch is not a dict, type(batch)={type(batch)}", {}
    u = ""
    if header is not None:
        u += f"=== {header} ===\n"
        u += get_separator("-") + "\n"

    if len(batch.keys()) == 0:
        return u + "Batch (Empty)", {}

    u += f"Batch (shape={infer_batch_shape(batch)}):\n"

    data = []
    exceptions = {}
    for k in sorted(batch.keys()):
        try:
            shape, leaf_type = infer_shape(batch[k], return_leaf_type=True)
            row = {
                "key": k,
                "shape": str(shape),
                "type": type(batch[k]).__name__,
                "leaf_type": str(leaf_type),
            }
            if report_nans:
                values = flatten_json_struct(batch[k])
                nan_count = sum(int(x is None) for x in values)
                row["nans"] = str(nan_count)

        except Exception as e:
            exceptions[k] = e
            row = {
                "key": "<error>",
                "shape": "<error>",
                "type": "<error>",
                "leaf_type": "<error>",
            }
            if report_nans:
                row["nans"] = "<error>"

        data += [row]

    keys = list(data[0].keys())
    column_width = {k: max([len(d[k]) for d in data]) for k in keys}
    newline_start = "  "
    column_sep = " [white]|[/white] " if rich else " | "

    # define a horizontal separator
    row_in = "[white]" if rich else ""
    row_out = "[/white]" if rich else ""
    seps = "".join(f"{'_' * m}" for m in column_width.values())
    horizontal_sep = f"{newline_start}{row_in}{seps}{row_out}\n"

    def format_row(row: Dict) -> str:
        u = newline_start
        u += column_sep.join(f"{row[k]:{column_width[k]}}" for k in keys)
        u += "\n"
        return u

    u += horizontal_sep
    u += format_row({k: k for k in keys})
    u += horizontal_sep
    for row in data:
        u += format_row(row)

    u += horizontal_sep

    return u, exceptions
