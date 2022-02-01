from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from transformers import PreTrainedTokenizerFast

from ...tokenizers.static import QUERY_MASK
from .base import Pipe
from .control.condition import HasPrefix
from .control.condition import In
from fz_openqa.modeling.functional import TorchBatch
from fz_openqa.utils.datastruct import Batch


class TokenizerPipe(Pipe):
    """tokenize a batch of data"""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        *,
        drop_columns: bool = True,
        fields: Union[str, List[str]],
        max_length: Optional[int],
        return_token_type_ids: bool = False,
        return_offsets_mapping: bool = False,
        add_special_tokens: bool = True,
        **kwargs,
    ):
        self.fields = [fields] if isinstance(fields, str) else fields
        assert kwargs.get("input_filter", None) is None, "input_filter is not allowed"
        super(TokenizerPipe, self).__init__(**kwargs, input_filter=In(self.fields))
        self.tokenizer = tokenizer
        self.drop_columns = drop_columns
        self.args = {
            "max_length": max_length,
            "truncation": max_length is not None,
            "return_token_type_ids": return_token_type_ids,
            "return_offsets_mapping": return_offsets_mapping,
            "add_special_tokens": add_special_tokens,
        }

    def output_keys(self, input_keys: List[str]) -> List[str]:
        if self.drop_columns:
            input_keys = []

        input_keys += ["input_ids", "attention_mask"]
        if self.args.get("return_offsets_mapping", False):
            input_keys += ["offset_mapping"]
        return input_keys

    def _call_batch(
        self, batch: TorchBatch, idx: Optional[List[int]] = None, **kwargs
    ) -> TorchBatch:
        tokenizer_input = {field: batch[field] for field in self.fields}

        batch_encoding = self.tokenizer(*tokenizer_input.values(), **self.args, **kwargs)

        if self.drop_columns:
            batch = {k: v for k, v in batch_encoding.items()}
        else:
            batch.update({k: v for k, v in batch_encoding.items()})

        return batch


class QueryExpansionPipe(Pipe):
    def __init__(
        self,
        *,
        prefix: Optional[str] = "question.",
        question_length: int = 32,
        tokenizer: PreTrainedTokenizerFast,
        **kwargs,
    ):
        if prefix is not None:
            input_filter = HasPrefix(prefix)
        else:
            input_filter = None
        super(QueryExpansionPipe, self).__init__(**kwargs, input_filter=input_filter)
        self.prefix = prefix if prefix is not None else ""
        self.question_length: int = question_length
        self.sep_token_id: int = tokenizer.sep_token_id
        self.q_mask_token_id: int = tokenizer.encode(QUERY_MASK, add_special_tokens=False)[0]

    @staticmethod
    def _last_idx(values: List, target) -> Optional[int]:
        if target in values:
            return len(values) - values[::-1].index(target) - 1
        else:
            return None

    def _insert_mapping(
        self, input_ids: List[int], attention_mask: List[int]
    ) -> Tuple[List[int], List[int]]:
        """insert the q_mask_token_id after the `sep_token` if available, else at the end"""
        if len(input_ids) >= self.question_length:
            return input_ids, attention_mask

        end_of_seq_idx = None
        if self.sep_token_id is not None:
            end_of_seq_idx = QueryExpansionPipe._last_idx(input_ids, self.sep_token_id)
        if end_of_seq_idx is None:
            if attention_mask[-1] == 0:
                end_of_seq_idx = attention_mask.index(0)
            else:
                end_of_seq_idx = len(input_ids) - 1

        # update the input_ids and the attention_mask, QMASK tokens are not masked out.
        input_ids = input_ids[:end_of_seq_idx] + [self.q_mask_token_id] * (
            self.question_length - end_of_seq_idx
        )
        attention_mask = attention_mask[:end_of_seq_idx] + [1] * (
            self.question_length - end_of_seq_idx
        )
        return input_ids, attention_mask

    def _call_batch(self, batch: Batch, idx: Optional[List[int]] = None, **kwargs) -> Batch:

        # pprint_batch(batch, type(self).__name__ + f"  mask: {self.q_mask_token_id}")

        input_ids_key = f"{self.prefix}input_ids"
        attention_mask_key = f"{self.prefix}attention_mask"

        new_input_ids = []
        new_attention_mask = []
        for ids, mask in zip(batch[input_ids_key], batch[attention_mask_key]):
            ids, mask = self._insert_mapping(ids, mask)
            new_input_ids += [ids]
            new_attention_mask += [mask]

        data = {
            input_ids_key: new_input_ids,
            attention_mask_key: new_attention_mask,
        }

        # pprint_batch(data, type(self).__name__ + " output")
        return data
