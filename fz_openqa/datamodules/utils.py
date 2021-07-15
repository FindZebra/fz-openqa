from typing import Any
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union


def gen_passages(
    sequence: List[int],
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
        _size = size - len(start_tokens)
        _stride = stride - len(start_tokens)
    else:
        start_tokens = []
        _size = size
        _stride = stride

    if end_tokens is not None:
        _size -= len(end_tokens)
        _stride -= len(end_tokens)
    else:
        end_tokens = []

    assert _size > 0
    assert _stride > 0
    assert _stride <= _size
    margin = _size - _stride
    for i in range(0, len(sequence), _stride):
        left_pad = margin // 2 + margin % 2 if i else 0
        right_pad = margin // 2
        center = _size - left_pad - right_pad
        seq = sequence[i : i + _size]
        padding = max(0, _size - len(seq)) if pad_token is not None else 0

        # only return if there are unmasked tokens
        if len(seq) > left_pad:
            seq = start_tokens + seq + end_tokens + padding * [pad_token]
            mask = (
                (len(start_tokens) + left_pad) * [0]
                + center * [1]
                + [0] * (len(end_tokens) + right_pad)
            )
            if padding > 0:
                mask[-padding:] = padding * [0]

            assert len(seq) == size, f"seq: {len(seq)}, size={size}"
            if return_mask:
                yield (
                    seq,
                    mask[: len(seq)],
                )
            else:
                yield seq
