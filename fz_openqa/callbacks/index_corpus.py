import shutil
from typing import Any
from typing import Optional
from typing import Sequence

import pytorch_lightning as pl
import rich
from pytorch_lightning.callbacks import BasePredictionWriter
from pytorch_lightning.callbacks import Callback

from fz_openqa.datamodules.corpus_dm import CorpusDataModule
from fz_openqa.utils.functional import infer_device


class IndexCorpus(Callback):
    # def __init__(self, write_interval: str = "batch") -> None:
    #     super().__init__(write_interval)

    def write_on_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        prediction: Any,
        batch_indices: Optional[Sequence[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Override with the logic to write a single batch."""
        print("=== write_on_batch_end ====")

    def write_on_epoch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        predictions: Sequence[Any],
        batch_indices: Optional[Sequence[Any]],
    ) -> None:
        """Override with the logic to write all batches."""
        print("=== write_on_epoch_end ====")

    def on_validation_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:

        # currently running trainer.predict() fucks up the trainning loop and should be avoided:
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/8158
        # moving the model weights back to the device seems to work, although there might be other issues
        # todo: reopen issue
        init_device = infer_device(pl_module)

        corpus: CorpusDataModule = trainer.datamodule.corpus
        rich.print(
            "--> step: {trainer.global_step}, rank: {trainer.global_rank}"
        )
        trainer.predict(model=pl_module, dataloaders=corpus.train_dataloader())
        rich.print("--> END")

        # temporary fix: move the weights back to the device
        pl_module.to(init_device)

    def display(self, input_ids, tokenizer):
        console_w, _ = shutil.get_terminal_size((100, 20))
        print(console_w * "-")
        for tokens in input_ids:
            print(tokenizer.decode(tokens).replace("[PAD]", ""))
        print(console_w * "-")
