from typing import List
from typing import Optional
from typing import Union

from transformers import PreTrainedTokenizerFast

from .base import Pipe
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
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.drop_columns = drop_columns
        self.fields = [fields] if isinstance(fields, str) else fields
        self.args = {
            "max_length": max_length,
            "truncation": max_length is not None,
            **kwargs,
        }

    def output_keys(self, input_keys: List[str]) -> List[str]:
        if self.drop_columns:
            input_keys = []

        input_keys += ["input_ids", "attention_mask"]
        if self.args.get("return_offsets_mapping", False):
            input_keys += ["offset_mapping"]
        return input_keys

    def _call(self, batch: TorchBatch, **kwargs) -> TorchBatch:
        tokenizer_input = {field: batch[field] for field in self.fields}

        batch_encoding = self.tokenizer(*tokenizer_input.values(), **self.args, **kwargs)

        if self.drop_columns:
            batch = {k: v for k, v in batch_encoding.items()}
        else:
            batch.update({k: v for k, v in batch_encoding.items()})

        return batch


class CleanupPadTokens(Pipe):
    """
    Remove pad tokens from all input_ids and corresponding attention_mask.
    Quick and ugly fix, sorry about that!
    """

    def __init__(self, tokenizer: PreTrainedTokenizerFast):
        self.pad_tok = tokenizer.pad_token_id

    def _call(self, batch: Batch, **kwargs) -> Batch:
        for k in batch.keys():
            if str(k).endswith(".input_ids"):
                all_tokens = batch[k]
                k_attn = k.replace(".input_ids", ".attention_mask")
                all_attn = batch[k_attn]
                assert isinstance(all_tokens, list), "not implemented for other types"

                # iterate examples
                new_tokens = []
                new_attn = []
                for i in range(len(all_tokens)):
                    tok_i = all_attn[i]
                    atn_i = all_attn[i]
                    if isinstance(tok_i[0], (list,)):
                        new_tokens_i = []
                        new_attn_i = []
                        for j in range(len(tok_i)):
                            tok_ij = tok_i[j]
                            atn_ij = atn_i[j]
                            tok_ij, atn_ij = self.filter_tokens(tok_ij, atn_ij)
                            new_tokens_i += [tok_ij]
                            new_attn_i += [atn_ij]
                        new_tokens += [new_tokens_i]
                        new_attn += [new_attn_i]
                    else:
                        tok_i, atn_i = self.filter_tokens(tok_i, atn_i)
                        new_tokens += [tok_i]
                        new_attn += [atn_i]

                batch[k] = new_tokens
                batch[k_attn] = new_attn

        return batch

    def filter_tokens(self, tokens, attn):
        tokens, atten = zip(*((t, a) for t, a in zip(tokens, attn) if t != self.pad_tok))
        return list(tokens), list(atten)
