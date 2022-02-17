from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import Optional

import rich
from torch import nn
from torch import Tensor
from transformers import BertConfig
from transformers import BertPreTrainedModel


class Head(nn.Module, ABC):
    """Score question and document representations."""

    id: str = "base"

    def __init__(self, *, bert_config: BertConfig, output_size: int, **kwargs):
        super(Head, self).__init__()

        self.input_size = bert_config.hidden_size
        self.output_size = output_size

    @abstractmethod
    def forward(
        self,
        *,
        hd: Tensor,
        hq: Tensor,
        doc_ids: Optional[Tensor] = None,
        q_mask: Optional[Tensor] = None,
        d_mask: Optional[Tensor] = None,
        batch: Dict[str, Tensor] = None,
        **kwargs
    ) -> Tensor:
        """
        Compute the score for each pair `f([q_j; a_j], d_jk)`.

        Parameters
        ----------
        hd
            Document representations of shape [bs, n_opts, n_docs, ...]
        hq
            Question representations of shape [bs, n_opts, ...]
        doc_ids
            Document ids of shape [bs, n_opts, n_docs]
        q_mask
            Mask for the question representations
        d_mask
            Mask for the document representations
        batch
            Batch of data
        Returns
        -------
        Tensor
            Scores of shape [bs, n_opts, n_docs]
        """
        raise NotImplementedError

    def preprocess(
        self, last_hidden_state: Tensor, head: str, mask: Optional[Tensor] = None
    ) -> Tensor:
        return last_hidden_state
