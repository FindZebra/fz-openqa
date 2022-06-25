from copy import copy
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from tokenizers import AddedToken
from transformers import PreTrainedTokenizerFast

from .base import Pipe
from fz_openqa.utils.datastruct import Batch


class ConcatTextFields(Pipe):
    """
    Concatenate two text fields
    """

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


DEFAULT_TOKEN_KEYS = [
    "input_ids",
    "attention_mask",
    "token_type_ids",
    "offsets_mapping",
]


class ConcatTokenFields(Pipe):
    """
    A pipe to concatenate multiple tokenized fields into one.
    """

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
        if end_if_truncated_tokens is None:
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


class ExtractGoldAnswer(Pipe):
    """
    Extract the gold answer from the answer options given the target
    """

    def __init__(
        self,
        *,
        answer_field: str = "answer",
        options_key: str = "answer",
        target_key: str = "target",
        output_key: str = "gold",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.answer_field = answer_field
        self.options_key = options_key
        self.target_key = target_key
        self.output_key = output_key

    def _call_batch(self, batch: Batch, **kwargs) -> Batch:
        options = batch[f"{self.answer_field}.{self.options_key}"]
        targets = batch[f"{self.answer_field}.{self.target_key}"]
        key = f"{self.answer_field}.{self.output_key}"
        return {key: [opts[t] for t, opts in zip(targets, options)]}


class InferTokenTypeIds(Pipe):
    def __init__(
        self,
        *,
        tokenizer: PreTrainedTokenizerFast,
        field: str = "question",
        symbol_map: List[List[str]] = None,
        **kwargs,
    ):
        """NB: when `symbol_map` is provided, all symbols not given
        in `symbol_map` will be ignored."""
        super(InferTokenTypeIds, self).__init__(**kwargs)
        self.field = field

        # register the special tokens
        special_tokens = [
            t.content if isinstance(t, AddedToken) else t
            for t in tokenizer.all_special_tokens_extended
        ]

        self.special_tokens = {
            t: tokenizer(t, add_special_tokens=False).input_ids[0] for t in special_tokens
        }

        # generate the symbol map or check the input
        if symbol_map is None:
            symbol_map = [[symbol] for symbol in self.special_tokens.keys()]
        else:
            for group in symbol_map:
                for symbol in group:
                    if symbol not in self.special_tokens:
                        raise ValueError(
                            f"symbol `{symbol}` is not a "
                            f"valid special token "
                            f"({list(self.special_tokens.keys())})"
                        )

        # register the symbol map {symbol : token_id}
        self.symbol_map = {symbol: i for i, group in enumerate(symbol_map) for symbol in group}

        # filter the `special_token` to include only the tokens registered in `symbol_map`
        self.special_tokens = {t: i for t, i in self.special_tokens.items() if t in self.symbol_map}

    def _call_batch(self, batch: Batch, **kwargs) -> Batch:
        token_type_ids = map(self.infer_token_types, batch[f"{self.field}.input_ids"])
        output = {f"{self.field}.token_type_ids": list(token_type_ids)}
        return output

    def infer_token_types(self, input_ids: List[int]) -> List[int]:
        # extract delimiters (special symbols with their position in `input_ids`)
        delimiters = {
            tok: input_ids.index(tok_id)
            for tok, tok_id in self.special_tokens.items()
            if tok_id in input_ids
        }
        delimiters = list(sorted(delimiters.items(), key=lambda x: x[1]))
        delimiters += [(None, len(input_ids))]

        # generate the token_type_ids from the delimiter positions
        token_type_ids = []
        for (symbol, start_pos), (next_tok, end_pos) in zip(delimiters[:-1], delimiters[1:]):
            symbol_id = self.symbol_map[symbol]
            segment = (end_pos - start_pos) * [symbol_id]
            token_type_ids.extend(segment)

        if len(token_type_ids) != len(input_ids):
            raise ValueError(
                f"`token_type_ids` and `input_ids` must have the same length. "
                f"Found {len(token_type_ids)} != {len(input_ids)}."
            )

        return token_type_ids
