import abc
import json
from typing import Union

from fz_openqa.datamodules.utils.datastruct import OpenQaDataset
from fz_openqa.datamodules.utils.typing import HfDataset


class Analytic:
    """
    The Analytic class allows to perform a specific analytic on a dataset.
    """

    def __init__(self, *, output_dir: str, verbose: bool = False):
        super().__init__()
        self.verbose = verbose
        self.output_dir = output_dir

    def save_as_json(self, results: Union[dict, list], filename: str) -> None:
        """
        Save results as json file.
        """
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)

    @abc.abstractmethod
    def __call__(self, dataset: Union[HfDataset, OpenQaDataset], **kwargs) -> None:
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
        pass
