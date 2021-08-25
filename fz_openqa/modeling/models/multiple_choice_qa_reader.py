import warnings
from typing import Dict
from typing import Tuple
from typing import Union

import rich
import torch
from omegaconf import DictConfig
from torch import nn
from torch import Tensor
from transformers import BertPreTrainedModel
from transformers import PreTrainedTokenizerFast

from fz_openqa.modeling.evaluators.base import BaseEvaluator
from fz_openqa.modeling.functional import flatten
from fz_openqa.modeling.functional import padless_cat
from fz_openqa.modeling.layers.heads import cls_head
from fz_openqa.modeling.models.base import BaseModel
from fz_openqa.utils.datastruct import pprint_batch


class MultipleChoiceQAReader(BaseModel):
    """
    A multiple-choice reader model.
    """

    _required_feature_names = [
        "question.input_ids",
        "question.attention_mask",
        "document.input_ids",
        "document.attention_mask",
        "answer.input_ids",
        "answer.attention_mask",
    ]
    # metrics that will be display in the progress bar
    _prog_bar_metrics = [
        "train/reader/loss",
        "validation/loss",
        "train/reader/Accuracy",
        "validation/reader/Accuracy",
    ]
    # prefix for the logged metrics
    _model_log_prefix = "reader/"

    def __init__(
        self,
        *,
        tokenizer: PreTrainedTokenizerFast,
        bert: Union[BertPreTrainedModel, DictConfig],
        evaluator: Union[BaseEvaluator, DictConfig],
        hidden_size: int = 256,
        dropout: float = 0,
        **kwargs,
    ):
        super().__init__(
            **kwargs, evaluator=evaluator, tokenizer=tokenizer, bert=bert
        )

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters(ignore=["evaluator", "tokenizer", "bert"])

        self.tokenizer = tokenizer  # todo

        # heads
        self.qd_head = cls_head(self.bert, hidden_size)
        self.qd_select_head = cls_head(self.bert, 1, normalize=False)
        self.a_head = cls_head(self.bert, hidden_size)
        self.dropout = nn.Dropout(dropout)

        # todo: infer maximum bert length
        self.max_length = 512  # self.bert.config.max_length

    def forward(
        self, batch: Dict[str, Tensor], **kwargs
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Compute the multiple choice answer model:
         * log p(a_i | q, e)
        where:
        *  log p(a_i| q, e) = Sim(BERT_CLS([q;d]), BERT_CLS(a_i))

        Future work:
        This is a very simple model, it will be improved in the next iterations.
        We can improve it either using a full interaction model
        * BERT_CLS([q;d;a_i])
        which requires roughly `N_q` times more GPU memory
        Or using a local interaction model `Sim` much similar to ColBERT.

        Input data:
        batch = {
        'document.input_ids': [batch_size * n_docs, L_d]
        'question.input_ids': [batch_size * n_docs, L_q]
        'answer.input_ids': [batch_size, N_q, L_q]
        }
        """
        # checks inputs, set parameters and concat the questions with the documents
        self.check_input_features(batch)
        bs, N_a, _ = batch["answer.input_ids"].shape
        # concatenate questions and documents such that there is no padding between Q and D
        padded_batch = self.concat_questions_and_documents(batch)

        # compute contextualized representations
        heq = self.bert(
            padded_batch["input_ids"][:, : self.max_length],
            padded_batch["attention_mask"][:, : self.max_length],
        ).last_hidden_state  # [bs*n_doc, L_e+L_q, h]

        ha = self.bert(
            flatten(batch["answer.input_ids"]),
            flatten(batch["answer.attention_mask"]),
        ).last_hidden_state  # [bs * N_a, L_a, h]

        # infer the number of documents
        n_docs = heq.shape[0] // bs
        assert n_docs > 0, (
            f"number of documents should be at least one: "
            f"got n_docs={n_docs}, heq.shape={heq.shape}, bs={bs}"
        )

        # selection model
        h_select = self.qd_select_head(self.dropout(heq)).mean(-1)
        h_select = h_select.view(bs, n_docs)

        # answer-question final representation
        heq = self.qd_head(self.dropout(heq))  # [bs * n_doc, h]
        ha = self.a_head(self.dropout(ha)).view(
            bs, N_a, self.hparams.hidden_size
        )
        # dot-product model S(qd, a)
        heq = heq.view(bs, n_docs, *heq.shape[1:])
        S_eqa = torch.einsum("bdh, bah -> bda", heq, ha)

        return S_eqa, h_select

    def concat_questions_and_documents(self, batch):
        """
        concatenate questions and documents such that
        there is no padding between Q and D
        """
        padded_batch = padless_cat(
            {
                "input_ids": batch["question.input_ids"],
                "attention_mask": batch["question.attention_mask"],
            },
            {
                "input_ids": batch["document.input_ids"],
                "attention_mask": batch["document.attention_mask"],
            },
            self.pad_token_id,
            aux_pad_tokens={"attention_mask": 0},
        )
        if len(padded_batch["input_ids"]) > self.max_length:
            warnings.warn("the tensor [question; document] was truncated.")

        return padded_batch

    @staticmethod
    def expand_and_flatten(x: Tensor, n: int) -> Tensor:
        """
        Expand a tensor of shape [bs, *dims] as
        [bs, n, *dims] and flatten to [bs * n, *dims]
        """
        bs, *dims = x.shape
        x = x[:, None].expand(bs, n, *dims)
        x = x.contiguous().view(bs * n, *dims)
        return x
