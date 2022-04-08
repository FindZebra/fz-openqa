from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import Optional
from typing import Union

from omegaconf import DictConfig
from torch import nn
from torch import Tensor
from transformers import AutoModel
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder
from transformers.models.bert.modeling_bert import BertPreTrainedModel

from fz_openqa.modeling.modules.utils.bert import instantiate_bert_model_with_config


class Head(nn.Module, ABC):
    """Score question and document representations."""

    id: str = "base"

    def __init__(
        self,
        *,
        bert: DictConfig | BertPreTrainedModel,
        output_size: int,
        split_bert_layers: int = 0,
        **kwargs
    ):
        super(Head, self).__init__()

        # instantiate bert
        bert = instantiate_bert_model_with_config(bert)

        self.input_size = bert.config.hidden_size
        self.output_size = output_size

        # use the last K layers of the BERT model
        if split_bert_layers == 0:
            self.bert_layers = None
        else:
            bert_encoder: BertEncoder = bert.encoder
            self.bert_layers = nn.ModuleList(bert_encoder.layer[-split_bert_layers:])

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
        if self.bert_layers is not None:
            last_hidden_state = self.bert_layers(last_hidden_state)

        return last_hidden_state

    def _preprocess(
        self, last_hidden_state: Tensor, head: str, mask: Optional[Tensor] = None
    ) -> Tensor:
        return last_hidden_state
