from typing import List
from typing import Optional
from typing import Union

from transformers import PreTrainedTokenizerFast

from ...modeling.functional import TorchBatch
from .base import Pipe


class TokenizerPipe(Pipe):
    """tokenize a batch of data"""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        *,
        drop_columns: bool = True,
        fields: Union[str, List[str]],
        max_length: Optional[int],
        **kwargs
    ):
        self.tokenizer = tokenizer
        self.drop_columns = drop_columns
        self.fields = [fields] if isinstance(fields, str) else fields
        self.args = {
            "max_length": max_length,
            "truncation": max_length is not None,
            **kwargs,
        }

    def __call__(self, batch: TorchBatch, **kwargs) -> TorchBatch:
        tokenizer_input = {field: batch[field] for field in self.fields}

        batch_encoding = self.tokenizer(
            *tokenizer_input.values(), **self.args, **kwargs
        )

        if self.drop_columns:
            batch = {k: v for k, v in batch_encoding.items()}
        else:
            batch.update({k: v for k, v in batch_encoding.items()})

        return batch
