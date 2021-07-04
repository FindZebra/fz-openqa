from typing import Any
from typing import Dict
from typing import Optional
from typing import Union

import torch
from omegaconf import DictConfig
from torch import nn
from torch import Tensor
from transformers import BertPreTrainedModel
from transformers import PreTrainedTokenizerFast

from fz_openqa.modeling.evaluators.abstract import Evaluator
from fz_openqa.modeling.layers.heads import cls_head
from fz_openqa.modeling.models.base import BaseModel


class QaRetriever(BaseModel):
    _required_infer_feature_names = [
        "question.input_ids",
        "question.attention_mask",
        "question.input_ids",
        "document.attention_mask",
        "question.input_ids",
        "question.attention_mask",
        "is_gold",
    ]
    _prog_bar_metrics = [
        "train/loss",
        "validation/loss",
        "train/Accuracy",
        "validation/Accuracy",
    ]  # metrics that will be display in the progress bar

    def __init__(
        self,
        *,
        tokenizer: PreTrainedTokenizerFast,
        bert: Union[BertPreTrainedModel, DictConfig],
        evaluator: Union[Evaluator, DictConfig],
        hidden_size: int = 256,
        dropout: float = 0,
        **kwargs,
    ):
        super().__init__(**kwargs, evaluator=evaluator)

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        # instantiate the pretrained model
        self.instantiate_bert(bert=bert, tokenizer=tokenizer)

        # projection head
        self.q_proj = cls_head(self.bert, hidden_size)
        self.e_proj = cls_head(self.bert, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        batch: Dict[str, Tensor],
        batch_idx: int = 0,
        dataloader_idx: int = None,
        model_key: str = "document",
        **kwargs,
    ) -> torch.Tensor:
        """Return the document/question representation."""
        assert model_key in {"document", "question"}, f"model_key={model_key}"
        input_ids = batch[f"{model_key}.input_ids"]
        attention_mask = batch.get(f"{model_key}.attention_mask", None)

        # compute contextualized representations
        h = self.bert(input_ids, attention_mask).last_hidden_state

        # global representations
        h = self.dropout(h)
        return {"document": self.e_proj, "question": self.q_proj}[model_key](h)

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Any:
        # compute contextualized representations
        h = self.bert(
            batch["input_ids"], batch["attention_mask"]
        ).last_hidden_state

        # global representations
        h = self.dropout(h)
        return {"document": self.e_proj, "question": self.q_proj}["document"](
            h
        )
