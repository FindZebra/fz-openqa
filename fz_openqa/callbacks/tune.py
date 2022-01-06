import logging
from typing import Dict
from typing import List
from typing import Union

from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneCallback

logger = logging.getLogger(__name__)


class TuneReportCallback(TuneCallback):
    """PyTorch Lightning to Ray Tune reporting callback

    Reports metrics to Ray Tune.

    Args:
        metrics (str|list|dict): Metrics to report to Tune. If this is a list,
            each item describes the metric key reported to PyTorch Lightning,
            and it will reported under the same name to Tune. If this is a
            dict, each key will be the name reported to Tune and the respective
            value will be the metric key reported to PyTorch Lightning.
        on (str|list): When to trigger checkpoint creations. Must be one of
            the PyTorch Lightning event hooks (less the ``on_``), e.g.
            "batch_start", or "train_end". Defaults to "validation_end".

    Example:

    .. code-block:: python

        import pytorch_lightning as pl
        from ray.tune.integration.pytorch_lightning import TuneReportCallback

        # Report loss and accuracy to Tune after each validation epoch:
        trainer = pl.Trainer(callbacks=[TuneReportCallback(
                ["val_loss", "val_acc"], on="validation_end")])

        # Same as above, but report as `loss` and `mean_accuracy`:
        trainer = pl.Trainer(callbacks=[TuneReportCallback(
                {"loss": "val_loss", "mean_accuracy": "val_acc"},
                on="validation_end")])

    """

    def __init__(
        self,
        metrics: Union[None, str, List[str], Dict[str, str]] = None,
        on: Union[str, List[str]] = "validation_end",
    ):
        super(TuneReportCallback, self).__init__(on)
        if isinstance(metrics, str):
            metrics = [metrics]
        self._metrics = metrics

    def _get_report_dict(self, trainer: Trainer, pl_module: LightningModule):
        # Don't report if just doing initial validation sanity checks.
        if trainer.sanity_checking:
            return
        if not self._metrics:
            report_dict = {k: v.item() for k, v in trainer.callback_metrics.items()}
        else:
            report_dict = {}
            for key in self._metrics:
                if isinstance(self._metrics, dict):
                    metric = self._metrics[key]
                else:
                    metric = key
                if metric in trainer.callback_metrics:
                    report_dict[key] = trainer.callback_metrics[metric].item()
                else:
                    logger.warning(
                        f"Metric {metric} does not exist in " "`trainer.callback_metrics."
                    )

        return report_dict

    def _handle(self, trainer: Trainer, pl_module: LightningModule):
        report_dict = self._get_report_dict(trainer, pl_module)
        if report_dict is not None:
            tune.report(**report_dict)
