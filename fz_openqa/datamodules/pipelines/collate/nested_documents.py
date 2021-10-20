import torch
from transformers import PreTrainedTokenizerFast

from fz_openqa.datamodules.pipes import AddPrefix
from fz_openqa.datamodules.pipes import ApplyToAll
from fz_openqa.datamodules.pipes import Collate
from fz_openqa.datamodules.pipes import FilterKeys
from fz_openqa.datamodules.pipes import FirstEg
from fz_openqa.datamodules.pipes import Flatten
from fz_openqa.datamodules.pipes import Gate
from fz_openqa.datamodules.pipes import Lambda
from fz_openqa.datamodules.pipes import Nest
from fz_openqa.datamodules.pipes import Parallel
from fz_openqa.datamodules.pipes import ReplaceInKeys
from fz_openqa.datamodules.pipes import Sequential
from fz_openqa.datamodules.utils.condition import HasKeyWithPrefix
from fz_openqa.datamodules.utils.filter_keys import KeyIn


class MaybeCollateDocuments(Gate):
    """
    Build a pipe to collate a batch of retrieved document.
    This is used only if documents are already stored with each q-a pair,
    which is the case when compiling datasets.
    """

    def __init__(
        self, n_documents: int, tokenizer: PreTrainedTokenizerFast, **kwargs
    ):

        # get the raw text
        raw_text_pipe = FilterKeys(
            KeyIn(["document.text", "document.match_on"])
        )

        # Get the simple attribute and cast to tensor
        simple_attr_pipe = Sequential(
            FilterKeys(
                KeyIn(
                    [
                        "document.idx",
                        "document.passage_idx",
                        "document.retrieval_score",
                        "document.is_positive",
                    ]
                )
            ),
            ApplyToAll(op=torch.tensor),
        )

        # collate the questions attributes (question.input_ids, question.idx, ...)
        tokens_pipe = Sequential(
            FilterKeys(
                KeyIn(["document.input_ids", "document.attention_mask"])
            ),
            ReplaceInKeys("document.", ""),
            Lambda(tokenizer.pad),
            AddPrefix("document."),
        )

        # the full pipe used to collate documents
        doc_collate_pipe = Sequential(
            Collate(
                keys=[
                    "document.idx",
                    "document.passage_idx",
                    "document.retrieval_score",
                    "document.match_on",
                    "document.is_positive",
                    "document.positive_count",
                    "document.input_ids",
                    "document.attention_mask",
                    "document.text",
                ]
            ),
            Flatten(),
            Parallel(raw_text_pipe, simple_attr_pipe, tokens_pipe),
            Nest(stride=n_documents),
        )

        condition = Sequential(FirstEg(), HasKeyWithPrefix("document."))

        super(MaybeCollateDocuments, self).__init__(
            condition, doc_collate_pipe, **kwargs
        )
