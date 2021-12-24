from __future__ import annotations  # noqa: F407

import itertools
from typing import Any
from typing import Dict
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from datasets import Dataset

from .base import Analytic


class PlotScoreDistributions(Analytic):
    """Plot the distribution of retrieval scores for matched documents"""

    requires_columns = ["document.retrieval_score", "document.match_score"]
    output_file_name = "score_distribution_plot"

    def process_dataset_split(self, dset: Dataset) -> List:
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
        labels = ["Positive"] * len(results["positives"]) + ["Negative"] * len(results["negatives"])
        scores = results["positives"] + results["negatives"]

        df = pd.DataFrame(
            {
                "Scores": scores,
                "Labels": labels,
            }
        )

        fig = sns.displot(
            df, x="Scores", hue="Labels", stat="density", common_norm=False, height=8, aspect=15 / 8
        )

        return fig

    @staticmethod
    def collate_splits(results: Dict) -> Dict:
        pos_scores = {"positives": val[0] for _, val in results.items()}
        neg_scores = {"negatives": val[1] for _, val in results.items()}
        return {**pos_scores, **neg_scores}

    def _process_results(self, results: Dict[str, Any]):
        """Process the results of the analytic."""
        results = self.collate_splits(results)
        fig = self.plot_split(results=results)
        self.save_as_png(fig)
