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

from fz_openqa.datamodules.corpus_dm import CorpusDataModule
from fz_openqa.modeling.evaluators.base import Evaluator
from fz_openqa.modeling.layers.heads import cls_head
from fz_openqa.modeling.pl_module import Module
from fz_openqa.utils.functional import maybe_instantiate


class QaRetriever(Module):
    """
    A Dense retriever model.
    """

    _required_feature_names = [
        "input_ids",
    ]
    # metrics that will be display in the progress bar
    _prog_bar_metrics = [
        "train/retriever/loss",
        "validation/retriever/loss",
        "validation/retriever/top5_Accuracy",
        "validation/retriever/top10_Accuracy",
    ]
    # prefix for the logged metrics
    _model_log_prefix = "retriever/"

    def __init__(
        self,
        *,
        tokenizer: PreTrainedTokenizerFast,
        bert: Union[BertPreTrainedModel, DictConfig],
        evaluator: Union[Evaluator, DictConfig],
        corpus: Optional[Union[CorpusDataModule, DictConfig]] = None,
        hidden_size: int = 256,
        dropout: float = 0,
        **kwargs,
    ):
        super().__init__(
            **kwargs, evaluator=evaluator, tokenizer=tokenizer, bert=bert
        )

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        # instantiate the pretrained model
        self.instantiate_bert(bert=bert, tokenizer=tokenizer)

        # save pointer to the corpus
        self.hparams.pop("corpus")
        self.corpus = None if corpus is None else maybe_instantiate(corpus)

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
        """
        compute the document and question representations based on the argument `model_key`.
        Return the representation of `x`: BERT(x)_CLS of shape [batch_size, h]

        Future work:
        Implement ColBERT interaction model, in that case the output
        will be `conv(BERT(x))` of shape [batch_size, T, h]
        """
        assert model_key in {"document", "question"}, f"model_key={model_key}"
        self.check_input_features_with_key(batch, model_key)

        # get input_ids and attention_mask
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
            batch["document.input_ids"],
            batch.get("document.attention_mask", None),
        ).last_hidden_state

        # global representations
        h = self.dropout(h)
        return {"document": self.e_proj, "question": self.q_proj}["document"](
            h
        )

    def check_input_features_with_key(self, batch, key):
        for f in self._required_feature_names:
            f = f"{key}.{f}"
            assert f in batch.keys(), f"The feature {f} is required."
