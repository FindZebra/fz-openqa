from typing import Union, Dict

import plotly.figure_factory as ff
from datasets import Dataset
from datasets import DatasetDict

import numpy as np

from ..utils.datastruct import OpenQaDataset
from ..utils.typing import HfDataset
from .base import Analytic
from fz_openqa.datamodules.utils.dataset import get_column_names


class PlotScoreDistributions(Analytic):
    """Count the number of questions matched with positive documents"""
    @staticmethod
    def plot_split(dset: Dataset) -> Dict:
        """
        Plot a specific split of the dataset.
        """
        n_scores = dset["document.retrieval_score"]
        scores = [np.round(y[0], 1) for y in n_scores]

        return scores

    def __call__(self, dataset: Union[HfDataset, OpenQaDataset], **kwargs):
        """Gather retrieval scores of positive documents and plot distribution."""
        assert "document.match_score" in get_column_names(dataset)
        if isinstance(dataset, DatasetDict):
            results = {split: self.plot_split(dset) for split, dset in dataset.items()}
        elif isinstance(dataset, Dataset):
            results = {"all": self.plot_split(dataset)}
        else:
            raise TypeError(f"Unsupported type {type(dataset)}")

        # create figure
        colors = ['rgb(241, 186, 195)', 'rgb(219, 83, 106)', 'rgb(125, 26, 43)']
        fig = ff.create_distplot(list(results.values()), list(results.keys()),
                                 colors=colors, bin_size=1, curve_type='normal')
        fig.update_yaxes(ticklabelposition="inside top", title="Density")
        fig.update_xaxes(title="Retrieval score")
        # log results
        self.save_as_html(fig, "score_distribution_plot.html")
