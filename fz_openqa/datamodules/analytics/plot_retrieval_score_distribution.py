import itertools
from typing import Dict
from typing import List
from typing import Union

import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import rich
from datasets import Dataset
from datasets import DatasetDict

from ..utils.datastruct import OpenQaDataset
from ..utils.typing import HfDataset
from .base import Analytic
from fz_openqa.datamodules.utils.dataset import get_column_names


class PlotScoreDistributions(Analytic):
    """Count the number of questions matched with positive documents"""

    @staticmethod
    def report_split(dset: Dataset) -> Dict:
        """
        Report a specific split of the dataset.
        """
        n_scores = dset["document.retrieval_score"]
        n_pos = dset["document.match_score"]
        pos_scores = [y[0].item() for x, y in zip(n_pos, n_scores) if sum(x) > 0]
        neg_scores = [y[1:].tolist() for x, y in zip(n_pos, n_scores) if sum(x) > 0]
        # flatten neg_scores
        neg_scores = list(itertools.chain(*neg_scores))

        return [pos_scores, neg_scores]

    @staticmethod
    def plot_split(results: Dict):
        """
        Plot for all splits in a dataset.
        """

        fig = go.Figure()
        colors = ["rgb(102, 178, 255)", "rgb(232, 97, 83)"]
        for i, (key, val) in enumerate(results.items()):
            fig.add_trace(
                go.Histogram(
                    name=key,
                    x=val,
                    histnorm="probability density",
                    marker_color=colors[i],
                    xbins=dict(start=0, end=250, size=5),
                    opacity=0.75,
                )
            )

        # Overlay both histograms
        fig.update_layout(barmode="overlay")
        fig.update_traces(hovertemplate=None, hoverinfo="skip")
        fig.update_yaxes(ticklabelposition="inside top", title="Density")
        fig.update_xaxes(title="Retrieval score")

        return fig

    @staticmethod
    def collate_splits(results: Dict) -> Dict:
        pos_scores = {"positives": val[0] for _, val in results.items()}
        neg_scores = {"negatives": val[1] for _, val in results.items()}
        return {**pos_scores, **neg_scores}

    def __call__(self, dataset: Union[HfDataset, OpenQaDataset], **kwargs):
        """Gather retrieval scores of positive documents and plot distribution."""
        assert "document.match_score" in get_column_names(dataset)
        if isinstance(dataset, DatasetDict):
            results = {split: self.report_split(dset) for split, dset in dataset.items()}
            collated_res = self.collate_splits(results=results)

        elif isinstance(dataset, Dataset):
            results = {"all": self.report_split(dataset)}
            collated_res = self.collate_splits(results=results)
        else:
            raise TypeError(f"Unsupported type {type(dataset)}")

        fig = self.plot_split(results=collated_res)

        # log results
        self.save_as_html(fig, "score_distribution_plot.html")
