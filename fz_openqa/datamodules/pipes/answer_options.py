from typing import List

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
