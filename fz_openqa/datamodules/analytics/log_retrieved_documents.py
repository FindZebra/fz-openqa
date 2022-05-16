from __future__ import annotations

import warnings
from io import StringIO
from typing import Any
from typing import Dict
from typing import List

import numpy as np
import rich
from datasets import Split
from rich.console import Console
from torch import Tensor

import wandb
from ..index._base import camel_to_snake
from ..utils.datastruct import OpenQaDataset
from ..utils.typing import HfDataset
from .base import Analytic
from .base import logger
from fz_openqa.utils.pretty import get_separator


def safe_cast_to_list(x: Any) -> Any:
    if isinstance(x, list):
        return [safe_cast_to_list(y) for y in x]
    elif isinstance(x, Tensor):
        if x.dim() == 0:
            return x.item()
        else:
            return [safe_cast_to_list(y) for y in x]


class LogRetrievedDocuments(Analytic):
    """Measure the accuracy of the retriever. Only works for concatenated [q; a] datasets."""

    requires_columns: List[str] = []
    output_file_name = "retrieved_documents.html"
    _allow_wandb: True
    n_samples: int = 10

    def __call__(self, dataset: HfDataset | OpenQaDataset, builder=None, **kwargs) -> None:
        """
        Log the documents retrieved in an OpenQA setting.
        """
        console = Console(record=True, file=None if self.verbose else StringIO())
        try:

            dset = dataset["train"]
            logger.info(f"Running {type(self).__name__} dataset of size {len(dset)}")
            collate_fn = builder.get_collate_pipe()
            row_ids = np.random.randint(0, len(dset), self.n_samples)
            batch = collate_fn([dset[int(i)] for i in row_ids], split=Split.TRAIN)
            total_repr = ""
            for i in range(self.n_samples):
                total_repr += get_separator("=")
                row = {k: v[i] for k, v in batch.items()}
                repr = builder.format_row(
                    row,
                )
                total_repr += repr

            # log to console
            console.print(total_repr)
            html = console.export_html()

            # log to wandb
            if self.wandb_log:
                try:
                    name = camel_to_snake(type(self).__name__)
                    wandb.log({name: wandb.Html(html, inject=False)}, commit=False)
                except Exception as e:
                    logger.warning(f"Could not log to wandb: {e}")

            # log to file
            logger.info(f"Saving analytics to {self.output_file_path.absolute()}")
            with open(self.output_file_path, "w") as f:
                f.write(html)
        except Exception as e:
            logger.error(f"Could not log retrieved documents: {e}")

    def _process_results(self, results: Dict[str, Any]):
        """Process the results of the analytic."""
        # log results
        dict_results = {k: r["dict"] for k, r in results.items()}
        self.save_as_json(dict_results)

        if self.verbose:
            print(get_separator())
            rich.print(f"=== {type(self).__name__}  ===")
            for split, r in results.items():
                print(get_separator("."))
                rich.print(f"{split}:")
                rich.print(r["str"])
            print(get_separator())

        if self.wandb_log:
            try:
                self.log_to_wandb(dict_results)
            except Exception as e:
                warnings.warn(f"Could not log to wandb: {e}")
