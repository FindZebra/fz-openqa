from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback

from fz_openqa.datamodules.__old.corpus_dm import CorpusDataModule


class AcceleratorWrapper:
    def __init__(self, trainer: pl.Trainer):
        self.trainer = trainer

    def __call__(self, batch: Any):
        batch["_mode_"] = "indexing"
        return self.trainer.accelerator.predict_step([batch, 0, None])


class IndexCorpus(Callback):
    # def __init__(self, write_interval: str = "batch") -> None:
    #     super().__init__(write_interval)

    @torch.no_grad()
    def on_validation_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """
        Compute the corpus vectors using the model.
        """
        corpus: CorpusDataModule = trainer.datamodule.corpus_dataset
        model = AcceleratorWrapper(trainer)
        corpus.build_index(model=model, index_mode="faiss")
