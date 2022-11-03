from copy import copy
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from warp_pipes import Batch
from warp_pipes import Pipe

DEFAULT_TOKEN_KEYS = [
    "input_ids",
    "attention_mask",
    "token_type_ids",
    "offsets_mapping",
]


class ConcatTextFields(Pipe):
    """Concatenate two text fields into a new one"""

    def __init__(
        self, keys: List[str], *, new_key: str = "concatenated", separator: str = ". ", **kwargs
    ):
        super().__init__(**kwargs)
        self.fields = keys
        self.separator = separator
        self.new_field = new_key

    def _call_batch(self, batch: Batch, **kwargs) -> Batch:
        columns = [batch[field] for field in self.fields]
        new_column = []
        for u in zip(*columns):
            new_column += [self.separator.join(u)]
        output = {self.new_field: new_column}
        return output

    def output_keys(self, input_keys: List[str]) -> List[str]:
        return [self.new_field]


class ConcatTokenFields(Pipe):
    """A pipe to concatenate multiple tokenized fields into one."""

    def __init__(
        self,
        fields: List[str],
        *,
        keys: Optional[List[str]] = None,
        master_key: str = "input_ids",
        new_field: str = "concatenated",
        drop_start_tokens: List[int] = None,
        keep_end_tokens: List[int] = None,
        max_length: Optional[int] = None,
        sep_tokens: Optional[Dict[str, Any]] = None,
        end_if_truncated_tokens: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fields = fields
        if keys is None:
            keys = DEFAULT_TOKEN_KEYS
        self.keys = keys
        assert master_key in keys
        self.master_key = master_key
        self.new_field = new_field
        if drop_start_tokens is None:
            drop_start_tokens = []
        self.drop_start_tokens = drop_start_tokens
        if keep_end_tokens is None:
            keep_end_tokens = []
        self.keep_end_tokens = keep_end_tokens
        self.max_length = max_length
        if sep_tokens is not None:
            sep_ = list(sep_tokens.values())[0]
            assert all(len(sep) == len(sep_) for sep in sep_tokens.values())
        self.sep_tokens = sep_tokens
        if end_if_truncated_tokens is not None:
            sep_ = list(sep_tokens.values())[0]
            assert all(len(sep) == len(sep_) for sep in end_if_truncated_tokens.values())
        self.end_if_truncated_tokens = end_if_truncated_tokens

    def _call_batch(self, batch: Batch, **kwargs) -> Batch:
        # validate the inputs keys
        master_field, *other_fields = self.fields
        keys = set(self.keys) & set([key.replace(f"{master_field}.", "") for key in batch.keys()])
        if self.sep_tokens is not None:
            assert set(self.sep_tokens.keys()) >= set(keys)
        if self.end_if_truncated_tokens is not None:
            assert set(self.end_if_truncated_tokens.keys()) >= set(keys)
        master_key, other_keys = self.master_key, list(set(keys) - {self.master_key})
        batch_size = len(batch[f"{master_field}.{master_key}"])

        # init the output with a copy of the master field
        output = {
            f"{self.new_field}.{key}": [copy(x) for x in batch[f"{master_field}.{key}"]]
            for key in keys
        }

        # concatenate the other fields with the master field, the master field
        # will be truncated to match `max_length` if it is set
        for i in range(batch_size):
            # find the number of beginning tokens to drop for each field (e.g. [CLS] token)
            neg_padding = {}
            length_other_fields = 0
            for field in other_fields:
                x = self.drop_start_tokens  # initial tokens to be removed
                y = batch[f"{field}.{master_key}"][i]
                neg_padding[field] = self.n_overlap(x, y)
                length_other_fields += len(y) - neg_padding[field]
                if self.sep_tokens is not None:
                    sep = self.sep_tokens[master_key]
                    length_other_fields += len(sep)

            # truncate the master field, insert the `end_if_truncated_tokens` when truncated
            # and complete with the `keep_end_tokens` (e.g. [SEP] token))
            if self.max_length is not None:
                master_length = len(output[f"{master_field}.{master_key}"][i])
                total_length = master_length + length_other_fields
                if total_length > self.max_length:
                    n_truncate = total_length - self.max_length
                    for key in keys:
                        x = self.keep_end_tokens  # initial tokens to be kept
                        y = output[f"{master_field}.{key}"][i]
                        n_keep_end = self.n_overlap(x[::-1], y[::-1])
                        m = n_truncate
                        if self.end_if_truncated_tokens is not None:
                            end_tokens = self.end_if_truncated_tokens[key]
                            m += len(end_tokens)
                        else:
                            end_tokens = []
                        if n_keep_end > 0:
                            y_pre = y[:-n_keep_end]
                            y_post = y[-n_keep_end:]
                        else:
                            y_pre = y
                            y_post = []
                        output[f"{master_field}.{key}"][i] = y_pre[:-m] + end_tokens + y_post

            # finally concatenate the other fields to the master field
            length_effectively_added = 0
            for field in other_fields:
                for key in (master_key, *other_keys):
                    y = copy(batch[f"{field}.{key}"][i])
                    if self.sep_tokens is not None:
                        sep = self.sep_tokens[key]
                        y = sep + y
                    y = y[neg_padding[field] :]
                    length_effectively_added += len(y)
                    output[f"{master_field}.{key}"][i] += y

            if self.max_length is not None:
                len_master = len(output[f"{master_field}.{master_key}"][i])
                if len_master > self.max_length:
                    raise ValueError(
                        f"The length of the master field {master_key} ({len_master}) "
                        f"is greater than {self.max_length}. "
                        f"Effectively added {length_effectively_added} tokens, "
                        f"planned for {length_other_fields}."
                    )

        return output

    @staticmethod
    def n_overlap(x: List, y: List) -> int:
        j = 0
        for i in range(min(len(x), len(y))):
            if x[i] == y[i]:
                j += 1
            else:
                break
        return j

    def output_keys(self, input_keys: List[str]) -> List[str]:
        return [self.new_field]
