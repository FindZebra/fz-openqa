from enum import Enum
from functools import partial
from typing import List
from typing import Optional

import rich
from transformers import PreTrainedTokenizer

from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.datamodules.pipes.utils.concatenate import concat_questions_and_documents
from fz_openqa.datamodules.pipes.utils.concatenate import stack_questions_and_documents
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.pretty import pprint_batch


class ViewFormat(Enum):
    """
    Enumeration of view formats.
    """

    READER_CONCAT = "reader_concat"
    READER_STACK = "reader_stack"


class View(Pipe):
    """This pipe allows preparing a batch of data to
    a specific target format."""

    def __init__(
        self,
        *,
        format: Optional[ViewFormat] = None,
        tokenizer: PreTrainedTokenizer = None,
        max_length: Optional[int] = None,
        output_field: Optional[str] = "qad",
        additional_columns: Optional[List[str]] = None,
        **kwargs,
    ):

        if format is not None:
            format = ViewFormat(format)
        self.format = format
        self.pad_token_id = tokenizer.pad_token_id
        self.max_length = max_length
        self.output_field = output_field
        self.additional_columns = additional_columns

        # define the view op
        args = {"max_length": self.max_length, "pad_token_id": self.pad_token_id}
        if self.format == ViewFormat.READER_CONCAT:
            self.op = partial(concat_questions_and_documents, **args)

        elif self.format == ViewFormat.READER_STACK:
            self.op = partial(stack_questions_and_documents, **args)

        else:
            raise ValueError(f"Unknown view format: {self.format}")

        super(View, self).__init__(**kwargs)

    def _call_batch(self, batch: Batch, idx: Optional[List[int]] = None, **kwargs) -> Batch:
        out = self.op(batch)
        out = {f"{self.output_field}.{k}": v for k, v in out.items()}
        for key in self.additional_columns:
            if key in batch.keys():
                out[key] = batch[key]
        return out
