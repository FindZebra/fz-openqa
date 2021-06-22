from typing import Dict, List, Any, Optional

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.distributions import Categorical
from torchmetrics.classification.accuracy import Accuracy
from transformers import PreTrainedTokenizerFast

from fz_openqa.modeling.evaluators.abstract import Evaluator


class LanguageModel(LightningModule):
    """
    Example of LightningModule for language modelling.
    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)
    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        *,
        tokenizer: PreTrainedTokenizerFast,
        evaluator: Evaluator,
        hidden_size: int = 256,
        max_length: int = 1000,
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

        # define the module here
        self.embeddings = nn.Embedding(
            self.vocabulary_size, self.vocabulary_size
        )
        self.rnn = nn.LSTM(
            input_size=self.vocabulary_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.proj = nn.Linear(hidden_size, self.vocabulary_size)

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

    def shift(self, batch: torch.Tensor) -> torch.Tensor:
        """right shift the input tensor across dim=1 and append 0 in front"""
        _zero = torch.zeros_like(batch[:, :1])
        return torch.cat([_zero, batch[:, :-1]], dim=1)

    def generate(
        self, n_samples: int = 10, max_length: Optional[int] = None
    ) -> torch.LongTensor:
        """Samplethe generative model autoregressively"""
        max_length = max_length or self.hparams.max_length
        device = next(iter(self.parameters())).device
        input_ids = torch.zeros(
            torch.Size([n_samples, 0]), dtype=torch.long, device=device
        )
        dummy = self.pad_token_id * torch.ones(
            torch.Size([n_samples, 1]), dtype=torch.long, device=device
        )
        stop_condition = False
        t = 0
        while not stop_condition:
            features = self.forward(torch.cat([input_ids, dummy], dim=1))
            sample_t = Categorical(logits=features[:, -1]).sample()
            input_ids = torch.cat([input_ids, sample_t[:, None]], dim=1)
            t += 1
            stop_condition = (
                t > max_length
                or (sample_t != self.pad_token_id).float().sum() == 0
            )

        return input_ids

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.FloatTensor:
        embs = self.embeddings(input_ids)
        logits, _ = self.rnn(self.shift(embs))
        logits = self.proj(logits)
        return logits

    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        data = self.evaluator(self.forward, batch)

        # log train metrics
        for k, v in data.items():
            self.log(
                f"train/{k}", v, on_step=False, on_epoch=True, prog_bar=False
            )

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return data

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        data = self.evaluator(self.forward, batch)

        # log train metrics
        for k, v in data.items():
            self.log(
                f"val/{k}", v, on_step=False, on_epoch=True, prog_bar=False
            )

        return data  # potentially add the other metrics here

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        data = self.evaluator(self.forward, batch)

        # log train metrics
        for k, v in data.items():
            self.log(
                f"test/{k}", v, on_step=False, on_epoch=True, prog_bar=False
            )

        return data  # potentially add the other metrics here

    def test_epoch_end(self, outputs: List[Any]):
        pass

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
