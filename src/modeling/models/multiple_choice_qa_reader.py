from typing import *

import torch
from datasets import Split
from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torchmetrics.classification.accuracy import Accuracy
from transformers import PreTrainedTokenizerFast, AutoModel, BertPreTrainedModel

from src.modeling.evaluators import Evaluator


def flatten(x: Tensor) -> Tensor:
    return x.view(-1, x.shape[-1])


class MultipleChoiceQAReader(LightningModule):
    _required_infer_feature_names = [
        'question.input_ids',
        'question.attention_mask',
        'question.input_ids',
        'document.attention_mask',
        'question.input_ids',
        'question.attention_mask',
        'answer_choices.input_ids',
        'answer_choices.attention_mask',
    ]
    _prog_bar_metrics = ['loss', 'accuracy']  # metrics that will be display in the progress bar

    def __init__(
            self,
            *,
            tokenizer: PreTrainedTokenizerFast,
            bert_id: str,
            evaluator: Evaluator,
            cache_dir: str,
            hidden_size: int = 256,
            lr: float = 0.001,
            weight_decay: float = 0.0005,
            **kwargs,
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.tokenizer = tokenizer
        self.vocabulary_size = len(tokenizer.get_vocab())
        self.pad_token_id = tokenizer.pad_token_id

        # evaluator: compute the loss and the metrics to be logged in
        self.evaluator = evaluator

        # pretrained model
        self.bert: BertPreTrainedModel = AutoModel.from_pretrained(bert_id, cache_dir=cache_dir)
        self.bert.resize_token_embeddings(len(tokenizer)) # necessary because of the added special tokens

        # projection heads
        self.e_proj = nn.Linear(self.bert.config.hidden_size, hidden_size)

        self.qa_attn = nn.MultiheadAttention(self.bert.config.hidden_size, self.bert.config.num_attention_heads)
        self.qa_proj = nn.Linear(self.bert.config.hidden_size, hidden_size)

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

    def e_repr(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        h = self.bert(input_ids, attention_mask).last_hidden_state
        h_cls = self.e_proj(h[:, 0])
        return h_cls

    def forward(self, batch: Dict[str, Tensor], **kwargs) -> torch.FloatTensor:
        """Compute the answer model p(a_i | q, e)"""
        for f in self._required_infer_feature_names:
            assert f in batch.keys(), f"The feature {f} is required for inference."

        # infer shapes
        bs, N_a, _ = batch['answer_choices.input_ids'].shape

        # compute contextualized representations
        he = self.bert(batch['document.input_ids'], batch['document.attention_mask']).last_hidden_state
        hq = self.bert(batch['question.input_ids'], batch['question.attention_mask']).last_hidden_state
        ha = self.bert(flatten(batch['answer_choices.input_ids']),
                       flatten(batch['answer_choices.attention_mask'])).last_hidden_state

        # evidence representation
        he_glob = self.e_proj(he[:, 0])

        # answer-question representation
        hq = self.expand_and_flatten(hq, N_a)
        hqa, _ = self.qa_attn(ha.permute(1, 0, 2), hq.permute(1, 0, 2), hq.permute(1, 0, 2))
        hqa_glob = self.qa_proj(hqa[0, :, :])

        # compute the interaction model (dot-product) and return as shape [bs, N_a]
        he_glob = self.expand_and_flatten(he_glob, N_a)
        logits = (hqa_glob * he_glob).sum(1)
        return logits.view(bs, N_a)

    @staticmethod
    def expand_and_flatten(x: Tensor, n: int) -> Tensor:
        """Expand a tensor of shape [bs, *dims] as [bs, n, *dims] and flatten to [bs * n, *dims]"""
        bs, *dims = x.shape
        x = x[:, None].expand(bs, n, *dims)
        x = x.contiguous().view(bs * n, *dims)
        return x

    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        data = self.evaluator(self.forward, batch, split=Split.TRAIN)

        # log train metrics
        for k, v in data.items():
            self.log(f"train/{k}", v, on_step=False, on_epoch=True, prog_bar=k in self._prog_bar_metrics)

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return data

    def validation_step(self, batch: Any, batch_idx: int):
        data = self.evaluator(self.forward, batch, split=Split.VALIDATION)

        # log train metrics
        for k, v in data.items():
            self.log(f"val/{k}", v, on_step=False, on_epoch=True, prog_bar=k in self._prog_bar_metrics)

        return data  # potentially add the other metrics here

    def test_step(self, batch: Any, batch_idx: int):
        data = self.evaluator(self.forward, batch, split=Split.TEST)

        # log train metrics
        for k, v in data.items():
            self.log(f"test/{k}", v, on_step=False, on_epoch=True, prog_bar=k in self._prog_bar_metrics)

        return data  # potentially add the other metrics here

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        metrics = self.evaluator.compute_metrics(split=Split.TRAIN)
        for k, v in metrics.items():
            self.log(k, v, prog_bar=k in self._prog_bar_metrics)
        self.evaluator.reset_metrics(split=Split.TRAIN)

    def validation_epoch_end(self, outputs: List[Any]):
        metrics = self.evaluator.compute_metrics(split=Split.VALIDATION)
        for k, v in metrics.items():
            self.log(k, v, prog_bar=k in self._prog_bar_metrics)
        self.evaluator.reset_metrics(split=Split.VALIDATION)

    def test_epoch_end(self, outputs: List[Any]):
        metrics = self.evaluator.compute_metrics(split=Split.TEST)
        for k, v in metrics.items():
            self.log(k, v, prog_bar=k in self._prog_bar_metrics)
        self.evaluator.reset_metrics(split=Split.TEST)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(
            params=self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
