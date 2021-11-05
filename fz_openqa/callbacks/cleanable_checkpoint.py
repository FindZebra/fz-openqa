import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


class CleanableCheckpoint(ModelCheckpoint):
    def __init__(self, *args, cleanup_threshold: str = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cleanup_threshold = cleanup_threshold

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_test_end(trainer, pl_module)
        # retrieve the checkpoint callback
        if self.best_model_score is not None:

            sign = {"max": 1, "min": -1}[self.mode]

            # delete checkpoint
            if sign * self.best_model_score < sign * self.cleanup_threshold:
                if os.path.exists(self.best_model_path):
                    self._del_model(self.best_model_path)
                if os.path.exists(self.last_model_path):
                    self._del_model(self.last_model_path)
