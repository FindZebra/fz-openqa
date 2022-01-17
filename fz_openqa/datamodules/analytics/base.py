from __future__ import annotations

import abc
import json
import logging
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Union

import plotly.graph_objects as go
import rich
import wandb
from datasets import Dataset
from datasets import DatasetDict

from fz_openqa.datamodules.index.base import camel_to_snake
from fz_openqa.datamodules.utils.dataset import get_column_names
from fz_openqa.datamodules.utils.datastruct import OpenQaDataset
from fz_openqa.datamodules.utils.typing import HfDataset
from fz_openqa.utils.pretty import get_separator

logger = logging.getLogger(__name__)


class Analytic:
    """
    The Analytic class allows to perform a specific analytic on a dataset.

    Attributes
    ----------
    requires_columns
        A list of columns that are required for the analytic to work.
    """

    requires_columns: List[str] = []
    output_file_name: str = "analytic.json"

    def __init__(self, *, output_dir: str, verbose: bool = False, wandb_log: bool = False):
        super().__init__()
        self.verbose = verbose
        self.wandb_log = wandb_log
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.output_file_path = self.output_dir / self.output_file_name

    def __call__(self, dataset: HfDataset | OpenQaDataset, **kwargs) -> None:
        """
        Perform the analytic on the dataset and report the results.

        Parameters
        ----------
        dataset
            The dataset to analyze.
        kwargs
            Additional optional arguments.

        Returns
        -------
        None

        """
        try:
            columns = get_column_names(dataset)
            for c in self.requires_columns:
                assert c in columns, (
                    f"{type(self).__name__} requires {self.requires_columns}. " f"Found {columns}"
                )

            if isinstance(dataset, DatasetDict):
                results = {
                    split: self.process_dataset_split(dset) for split, dset in dataset.items()
                }
            elif isinstance(dataset, Dataset):
                results = {"all": self.process_dataset_split(dataset)}
            else:
                raise TypeError(f"Unsupported type {type(dataset)}")

            return self._process_results(results)
        except Exception:
            logger.exception(f"Error while processing {type(self).__name__}")

    @abc.abstractmethod
    def process_dataset_split(self, dset: Dataset) -> Dict | List:
        """Process and report on a specific split of the dataset."""
        raise NotImplementedError

    def _process_results(self, results: Dict[str, Any]):
        """Process the results of the analytic."""
        # log results
        self.save_as_json(results)

        if self.verbose:
            self.pprint_json_results(results)

        if self.wandb_log:
            self.log_to_wandb(results)

    def save_as_json(self, results: Union[dict, list]) -> None:
        """
        Save results as json file.
        """
        logging.info(f"Saving analytics to {self.output_file_path.absolute()}")
        with open(self.output_file_path, "w") as f:
            json.dump(results, f, indent=2)

    def save_as_html(self, fig: go.Figure) -> None:
        """
        Save plot as html file
        """
        logging.info(f"Saving analytics to {self.output_file_path.absolute()}")
        fig.write_html(self.output_file_path, include_plotlyjs=True)

    def pprint_json_results(self, results: List | Dict):
        """Print results as JSON"""
        print(get_separator())
        rich.print(f"=== {type(self).__name__} ===")
        print(get_separator("."))
        rich.print(json.dumps(results, indent=2))
        print(get_separator())

    def log_to_wandb(self, results: List | Dict):
        """Log results to wandb"""
        name = camel_to_snake(type(self).__name__)
        try:
            self._log_leaf_to_wandb(name, results)
        except Exception as e:
            logger.warning(f"Could not log to wandb: {e}")

    @staticmethod
    def _log_leaf_to_wandb(key, value):
        if isinstance(value, dict):
            for k, v in value.items():
                Analytic._log_leaf_to_wandb(f"{key}/{k}", v)

        else:
            wandb.log({key: value})
