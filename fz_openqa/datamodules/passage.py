from typing import Any
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union


def gen_passages(
    sequence: List[Any],
    *,
    size: int,
    stride: int,
    start_tokens: Optional[List[Any]] = None,
    end_tokens: Optional[List[Any]] = None,
    pad_token: Optional[Any] = None,
    return_mask: bool = True,
) -> Iterable[Union[List[int], Tuple[List[int], List[Any]]]]:
    """Generate overlapping windows with the corresponding masking such that each token appears only in one window."""

    if start_tokens is not None:
        eff_size = size - len(start_tokens)
        eff_stride = stride - len(start_tokens)
    else:
        start_tokens = []
        eff_size = size
        eff_stride = stride

    if end_tokens is not None:
        eff_size -= len(end_tokens)
        eff_stride -= len(end_tokens)
    else:
        end_tokens = []

    assert eff_size > 0
    assert eff_stride > 0
    assert eff_stride <= eff_size
    margin = eff_size - eff_stride
    for i in range(0, len(sequence), eff_stride):
        left_pad = margin // 2 + margin % 2 if i else 0
        right_pad = margin // 2
        center = eff_size - left_pad - right_pad
        seq = sequence[i : i + eff_size]
        padding = max(0, eff_size - len(seq)) if pad_token is not None else 0

        # only return if there are unmasked tokens
        if len(seq) > left_pad:

            # define the passage
            seq = start_tokens + seq + end_tokens + padding * [pad_token]

            # define the passage mask
            mask = (
                (len(start_tokens) + left_pad) * [0]
                + center * [1]
                + [0] * (len(end_tokens) + right_pad)
            )
            if padding > 0:
                mask[-padding:] = padding * [0]

            if return_mask:
                yield (
                    seq,
                    mask[: len(seq)],
                )
            else:
                yield seq
