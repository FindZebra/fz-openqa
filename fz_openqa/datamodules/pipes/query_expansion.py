from typing import List
from typing import Optional
from typing import Tuple

from transformers import PreTrainedTokenizerFast
from warp_pipes import Batch
from warp_pipes import HasPrefix
from warp_pipes import Pipe

from fz_openqa.transformers_utils.tokenizer import QUERY_MASK


class QueryExpansion(Pipe):
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
        super(QueryExpansion, self).__init__(**kwargs, input_filter=input_filter)
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
            end_of_seq_idx = QueryExpansion._last_idx(input_ids, self.sep_token_id)
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
