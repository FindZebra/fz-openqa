import warnings
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

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


class MultipleChoiceQAReader(BaseModel):
    _required_infer_feature_names = [
        "question.input_ids",
        "question.attention_mask",
        "document.input_ids",
        "document.attention_mask",
        "answer_choices.input_ids",
        "answer_choices.attention_mask",
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
        cache_dir: Optional[str] = None,
        hidden_size: int = 256,
        dropout: float = 0,
        **kwargs,
    ):
        super().__init__(**kwargs, evaluator=evaluator)

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        # instantiate the pretrained model
        self.instantiate_bert(
            bert=bert, tokenizer=tokenizer, cache_dir=cache_dir
        )

        # heads
        self.qd_head = cls_head(self.bert, hidden_size)
        self.qd_select_head = cls_head(self.bert, 1, normalize=False)
        self.a_head = cls_head(self.bert, hidden_size)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, batch: Dict[str, Tensor], **kwargs
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Compute the answer model p(a_i | q, e)"""
        for f in self._required_infer_feature_names:
            assert (
                f in batch.keys()
            ), f"The feature {f} is required for inference."

        # infer shapes
        bs, N_a, _ = batch["answer_choices.input_ids"].shape

        # concatenate questions and documents such that there is no padding between Q and D
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
        # compute contextualized representations
        if len(padded_batch["input_ids"]) > 512:
            warnings.warn("the tensor [question; document] was truncated.")
        heq = self.bert(
            padded_batch["input_ids"][:, :512],
            padded_batch["attention_mask"][:, :512],
        ).last_hidden_state  # [bs*n_doc, L_e+L_q, h]
        ha = self.bert(
            flatten(batch["answer_choices.input_ids"]),
            flatten(batch["answer_choices.attention_mask"]),
        ).last_hidden_state  # [bs * N_a, L_a, h]

        # answer-question representation
        h_select = self.qd_select_head(self.dropout(heq))
        heq = self.qd_head(self.dropout(heq))  # [bs * n_doc, h]
        ha = self.a_head(self.dropout(ha)).view(
            bs, N_a, self.hparams.hidden_size
        )
        # dot-product model
        n_docs = heq.shape[0] // ha.shape[0]
        heq = heq.view(bs, n_docs, *heq.shape[1:])
        S_eqa = torch.einsum("bdh, bah -> bda", heq, ha)

        # return S(qd, a) and the selection model
        h_select = h_select.view(bs, n_docs)
        return S_eqa, h_select

    @staticmethod
    def expand_and_flatten(x: Tensor, n: int) -> Tensor:
        """Expand a tensor of shape [bs, *dims] as [bs, n, *dims] and flatten to [bs * n, *dims]"""
        bs, *dims = x.shape
        x = x[:, None].expand(bs, n, *dims)
        x = x.contiguous().view(bs * n, *dims)
        return x
