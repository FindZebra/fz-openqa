import torch
from torch import Tensor, nn
from transformers import PreTrainedTokenizerFast, AutoModel, BertPreTrainedModel

from src.modeling.evaluators import Evaluator
from .base import BaseModel


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
        bert_id: str,
        evaluator: Evaluator,
        cache_dir: str,
        hidden_size: int = 256,
        dropout: float = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.tokenizer = tokenizer
        self.vocabulary_size = len(tokenizer.get_vocab())
        self.pad_token_id = tokenizer.pad_token_id

        # evaluator: compute the loss and the metrics to be logged in
        self.evaluator = evaluator

        # pretrained model
        self.bert: BertPreTrainedModel = AutoModel.from_pretrained(
            bert_id, cache_dir=cache_dir
        )
        self.bert.resize_token_embeddings(
            len(tokenizer)
        )  # necessary because of the added special tokens

        # projection head
        self.q_proj = nn.Linear(self.bert.config.hidden_size, hidden_size)
        self.e_proj = nn.Linear(self.bert.config.hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, *, input_ids: Tensor, attention_mask: Tensor, key: str, **kwargs
    ) -> torch.FloatTensor:
        """Return the document/question representation."""
        assert key in {"document", "question"}

        # compute contextualized representations
        h = self.bert(input_ids, attention_mask).last_hidden_state

        # global representations
        h = self.dropout(h)
        h_glob: Tensor = {"document": self.e_proj, "question": self.q_proj}[key](h)

        # todo: masking
        return torch.nn.functional.normalize(h_glob, p=2, dim=2)
