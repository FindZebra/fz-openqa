import pytorch_lightning as pl
from pympler import muppy
from pympler import summary
from pytorch_lightning import Callback


class PymplerCallback(Callback):
    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        all_objects = muppy.get_objects()
        sum1 = summary.summarize(all_objects)
        # Prints out a summary of the large objects
        summary.print_(sum1)
