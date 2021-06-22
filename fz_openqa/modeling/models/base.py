from typing import Dict, List, Any

from datasets import Split
from pytorch_lightning import LightningModule
from transformers import AdamW


class BaseModel(LightningModule):
    _required_infer_feature_names = []
    _prog_bar_metrics = []  # metrics that will be display in the progress bar

    def __init__(
        self,
        *,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        **kwargs,
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

    def _step(
        self, batch: Any, batch_idx: int, split: Split
    ) -> Dict[str, Any]:
        data = self.evaluator(self.forward, batch, split=split)

        # log 'split' metrics
        for k, v in data.items():
            self.log(
                f"{split}/{k}",
                v,
                on_step=False,
                on_epoch=True,
                prog_bar=f"{split}/{k}" in self._prog_bar_metrics,
                sync_dist=True,
            )

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return data

    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        return self._step(batch, batch_idx, Split.TRAIN)

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        return self._step(batch, batch_idx, Split.VALIDATION)

    def test_step(self, batch: Any, batch_idx: int):
        return self._step(batch, batch_idx, Split.VALIDATION)

    def _epoch_end(self, outputs: List[Any], split: Split):
        # `outputs` is a list of dicts returned from `training_step()`
        metrics = self.evaluator.compute_metrics(split=split)
        for k, v in metrics.items():
            self.log(
                k, v, prog_bar=k in self._prog_bar_metrics, sync_dist=True
            )
        self.evaluator.reset_metrics(split=split)

    def training_epoch_end(self, outputs: List[Any]):
        return self._epoch_end(outputs, Split.TRAIN)

    def validation_epoch_end(self, outputs: List[Any]):
        return self._epoch_end(outputs, Split.VALIDATION)

    def test_epoch_end(self, outputs: List[Any]):
        return self._epoch_end(outputs, Split.TEST)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return AdamW(
            params=self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
