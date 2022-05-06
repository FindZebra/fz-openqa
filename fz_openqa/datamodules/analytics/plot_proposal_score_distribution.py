from __future__ import annotations

import itertools
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

from datasets import Dataset
from datasets import Split

from .base import Analytic


class PlotScoreDistributions(Analytic):
    """Plot the distribution of retrieval scores for matched documents"""

    requires_columns = ["document.proposal_score", "document.match_score"]
    output_file_name = "score_distribution_plot.html"

    def process_dataset_split(
        self, dset: Dataset, *, split: Optional[str | Split] = None
    ) -> Dict | List:
        """
        Report a specific split of the dataset.
        """
        n_scores = dset["document.proposal_score"]
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

    def _process_results(self, results: Dict[str, Any]):
        """Process the results of the analytic."""
        results = self.collate_splits(results)
        fig = self.plot_split(results=results)
        self.save_fig_as_html(fig)
