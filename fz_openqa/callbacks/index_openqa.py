from typing import Optional

import pytorch_lightning as pl
import rich
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.connectors.data_connector import DataConnector
from warp_pipes import get_console_separator

from fz_openqa.datamodules import DataModule


class IndexOpenQaCallback(Callback):
    """A Callback to re-index the OpenQA dataset."""

    datamodule: Optional[DataModule] = None

    def __init__(
        self,
        skip_first_epoch: bool = True,
        frequency: int = 1,
        keep_in_memory: bool = True,
        **kwargs,
    ) -> None:
        """

        Parameters
        ----------
        skip_first_epoch
            If True, the first epoch will be skipped.
        frequency
            The frequency at which the dataset will be updated.
        keep_in_memory
            If True, the dataset will be kept in memory.
        kwargs
            Keyword arguments passed to `Callback`.
        """
        super().__init__(**kwargs)
        self.skip_first_epoch = skip_first_epoch
        self.frequency = frequency
        self.keep_in_memory = keep_in_memory

    def attach(self, datamodule: DataModule) -> None:
        self.datamodule = datamodule

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Re-index the OpenQA dataset."""

        if self.skip_first_epoch and trainer.current_epoch == 0:
            return

        if trainer.current_epoch % self.frequency != 0:
            return

        if self.datamodule is None:
            raise ValueError(
                "You must first attach this callback to a DataModule. "
                "Usage: `callback.attach(datamodule)`"
            )

        # get the data connector
        data_connector: DataConnector = trainer._data_connector
        datamodule = trainer.datamodule

        print(f"\n{get_console_separator()}\n")
        rich.print(
            f">>>> [cyan]{type(self).__name__}[/cyan]: epoch={trainer.current_epoch}\n"
            f"trainer._is_data_prepared: {trainer._is_data_prepared}\n"
            f"data_connector:"
        )
        rich.print(data_connector)
        rich.print(f"> train_data_fetcher: {data_connector.train_data_fetcher}")
        print(f"\n{get_console_separator()}\n")

        # cleanup the data connector
        # data_connector._train_dataloader_source = None
        # data_connector._val_dataloader_source = None
        # data_connector._test_dataloader_source = None
        # data_connector._predict_dataloader_source = None

        # update the dataset
        rich.print(f"[magenta]{type(self).__name__}[/magenta]: epoch={trainer.current_epoch}")
        datamodule.setup(model=pl_module, trainer=trainer, keep_in_memory=self.keep_in_memory)

        # make sure to re-attach the datamodule to the `pl.Trainer`
        # data_connector.attach_datamodule(pl_module, datamodule=self.datamodule)

        print(f"\n{get_console_separator()}\n")
        rich.print(
            f">>>> [magenta]{type(self).__name__}[/magenta]: epoch={trainer.current_epoch}\n"
            f"trainer._is_data_prepared: {trainer._is_data_prepared}\n"
            f"data_connector:"
        )
        rich.print(data_connector)
        rich.print(f"> train_data_fetcher: {data_connector.train_data_fetcher}")
        print(f"\n{get_console_separator()}\n")
