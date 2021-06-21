from typing import List, Optional, Any, Iterable, Union, Tuple


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
        size -= len(start_tokens)
        stride -= len(start_tokens)
    else:
        start_tokens = []

    if end_tokens is not None:
        size -= len(end_tokens)
        stride -= len(end_tokens)
    else:
        end_tokens = []

    assert size > 0
    assert stride > 0
    assert stride <= size
    margin = size - stride
    for i in range(0, len(sequence), stride):
        left_pad = margin // 2 + margin % 2 if i else 0
        right_pad = margin // 2
        center = size - left_pad - right_pad
        seq = sequence[i: i + size]
        padding = max(0, size - len(seq)) if pad_token is not None else 0

        # only return if there are unmasked tokens
        if len(seq) > left_pad:
            seq = start_tokens + seq + end_tokens + padding * [pad_token]
            mask = (len(start_tokens) + left_pad) * [0] + center * [1] + [0] * (len(end_tokens) + right_pad)
            if padding > 0:
                mask[-padding:] = padding * [0]
            if return_mask:
                yield (
                    seq,
                    mask[: len(seq)],
                )
            else:
                yield seq