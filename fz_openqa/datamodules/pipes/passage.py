from collections import defaultdict
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from .base import Pipe
from fz_openqa.utils.datastruct import Batch


class GeneratePassages(Pipe):
    """A pipe to Extract passages from a batch of documents."""

    required_keys = ["input_ids", "attention_mask", "offset_mapping"]

    def __init__(
        self,
        *,
        size: int,
        stride: int,
        start_tokens: List[int],
        end_tokens: List[int],
        pad_token_id: int,
        verbose: bool = True,
        **kwargs,
    ):
        super(GeneratePassages, self).__init__(**kwargs)

        self.verbose = verbose
        base_args = {"size": size, "stride": stride}
        self.args = {
            "input_ids": {
                "pad_token": pad_token_id,
                "start_tokens": start_tokens,
                "end_tokens": end_tokens,
                **base_args,
            },
            "attention_mask": {
                "pad_token": 0,
                "start_tokens": [0 for _ in start_tokens],
                "end_tokens": [0 for _ in end_tokens],
                **base_args,
            },
            "offset_mapping": {
                "pad_token": [-1, -1],
                "start_tokens": [[-1, -1] for _ in start_tokens],
                "end_tokens": [[-1, -1] for _ in end_tokens],
                **base_args,
            },
        }

    def output_keys(self, input_keys: List[str]) -> List[str]:
        return input_keys + ["idx", "passage_idx", "passage_mask"]

    def _call(self, batch: Batch, **kwargs) -> Batch:
        self._check_input_keys(batch)
        indexes, output = self.generate_passages_for_all_keys(
            batch,
            keys=["input_ids", "attention_mask", "offset_mapping"],
            args=self.args,
        )

        # extract document.text
        output["text"] = [
            self.extract_passage_text_from_doc(batch["text"][idx], ofs_ids)
            for idx, ofs_ids in zip(indexes, output["offset_mapping"])
        ]

        return output

    def _check_input_keys(self, batch):
        for key in self.required_keys:
            assert key in batch.keys(), (
                f"key={key} must be provided. " f"Found batch.keys={list(batch.keys())}."
            )

    @staticmethod
    def generate_passages_for_all_keys(
        examples: Dict[str, List[Any]],
        keys: List[str],
        args: Dict[str, Dict[str, Any]],
    ) -> Tuple[List[int], Batch]:
        """
        This functions generate the passages for each attribute in `keys`,
         the `arg` dictionary must contain an entry for all `keys`.
         The first pass is used to store the document/example indexes
        and compute the `passage_mask`.

        The passage mask is used for segmentation, and is optional for this project.
        In this context, all tokens are attributed to a single passage,
        although they appear in multiple passages (strides).
        The passage mask indicates if a token is attributed to this specific passage.

        return:
            - indexes: index of the parent example for each passage
            - output: Batch of data for all keys + `idx` (document id) and `passage_mask`
        """
        assert "idx" in examples.keys()
        assert all(key in args.keys() for key in keys)
        L = len(next(iter(examples.values())))
        assert all(L == len(x) for x in examples.values())

        first_key, *other_keys = keys
        output = defaultdict(list)
        indexes = []
        for idx, (doc_idx, example) in enumerate(zip(examples["idx"], examples[first_key])):

            # do a first pass to compute the passage masks
            for pas_idx, (passage, passage_mask) in enumerate(
                gen_passages(example, **args[first_key], return_mask=True)
            ):
                indexes += [idx]
                output["idx"].append(doc_idx)
                output["passage_idx"].append(pas_idx)
                output["passage_mask"].append(passage_mask)
                output[first_key].append(passage)

            # do another pass to generate the passages for each remaining attribute
        for key in other_keys:
            for example in examples[key]:
                passages = gen_passages(example, **args[key], return_mask=False)
                for i, passage in enumerate(passages):
                    output[key].append(passage)

        # check output consistency and return
        L = len(list(next(iter(output.values()))))
        assert all(len(v) == L for v in output.values())
        return indexes, output

    @staticmethod
    def extract_passage_text_from_doc(document: str, offset_mapping: List[Tuple[int, int]]) -> str:
        """
        Extract the text passage from the original document
        given the offset mapping of the passage
        """
        indexes = [x for idxes_tok in offset_mapping for x in idxes_tok if x >= 0]
        return document[min(indexes) : max(indexes)]


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
    """Generate overlapping windows with the corresponding
    masking such that each token appears only in one window."""

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
