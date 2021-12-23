from __future__ import annotations  # noqa: F407

import functools
import itertools
from collections import Counter
from typing import Dict
from typing import List

import pandas as pd
import plotly.express as px
import rich
import seaborn as sns
from datasets import Dataset

from .base import Analytic
from fz_openqa.utils.pretty import get_separator


class PlotTopMatchTriggers(Analytic):
    """ Plot the top 20 triggers for matched documents"""

    requires_columns = ["document.retrieval_score", "document.match_score"]
    output_file_name = "top_match_triggers"

    def __init__(self, *args, topn: int = 100, **kwargs):
        super(PlotTopMatchTriggers, self).__init__(*args, **kwargs)
        self.topn = topn

    def process_dataset_split(self, dset: Dataset) -> Counter:
        """
        Report a specific split of the dataset.
        """
        n_triggers = dset["document.match_on"]
        n_pos = dset["document.match_score"]

        # Extract triggers for only positive matches
        n_triggers = [y[0] for x, y in zip(n_pos, n_triggers) if sum(x) > 0]
        # Flatten list
        triggers = list(itertools.chain(*n_triggers))

        # Counter returns a dict for counting hashable objects
        return Counter(triggers)

    def plot_most_commons(self, labels: List[str], counts: List[int]):
        """
        Plot most common values.
        """

        fig = px.bar(y=counts, x=labels, text=counts)
        fig.update_traces(textposition="outside", marker_color="rgb(102, 178, 255)")
        fig.update_yaxes(ticklabelposition="inside top", title="Count")
        fig.update_xaxes(title="Triggers")

        return fig

    @staticmethod
    def collate_splits(results: Dict[str, Counter]) -> Counter:
        """ Collate results of splits """
        rich.print(results)
        return functools.reduce(lambda x, y: x + y, results.values())

    def _process_results(self, results: Dict):
        """Process the results of the analytic."""
        counter = self.collate_splits(results)

        labels, counts = zip(*counter.most_common(self.topn))
        if self.verbose:
            print(get_separator())
            rich.print(f"=== {type(self).__name__} ===")
            print(get_separator("."))
            c_length = max(len(str(c)) for c in counts)
            for label, count in zip(labels, counts):
                rich.print(f" count={str(count):{c_length}} label='{label}'")
            print(get_separator())

        fig = self.plot_most_commons(labels, counts)
        self.save_as_html(fig)
