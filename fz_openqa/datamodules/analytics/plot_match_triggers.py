import itertools
from collections import Counter
from typing import Dict
from typing import List
from typing import Union

import plotly.express as px
import rich
from datasets import Dataset
from datasets import DatasetDict

from ..utils.datastruct import OpenQaDataset
from ..utils.typing import HfDataset
from .base import Analytic
from fz_openqa.datamodules.utils.dataset import get_column_names


class PlotTop20MatchTriggers(Analytic):
    """Count the number of questions matched with positive documents"""

    @staticmethod
    def report_split(dset: Dataset) -> Dict:
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
        count_triggers = Counter(triggers)

        return count_triggers

    @staticmethod
    def plot_splits(label: List[str], count: List[int]):
        """
        Plot for all splits in a dataset.
        """
        # create figure
        fig = px.bar(y=count, x=label, text=count)
        fig.update_traces(textposition="outside", marker_color="rgb(102, 178, 255)")
        fig.update_yaxes(ticklabelposition="inside top", title="Count")
        fig.update_xaxes(title="Triggers")

        return fig

    @staticmethod
    def collate_results(train, val, test):
        """ Collate results of splits """
        counter = train + val + test
        return counter.most_common(20)

    def __call__(self, dataset: Union[HfDataset, OpenQaDataset], **kwargs):
        """Gather retrieval scores of positive documents and plot distribution."""
        assert "document.match_score" in get_column_names(dataset)
        if isinstance(dataset, DatasetDict):
            results = {split: self.report_split(dset) for split, dset in dataset.items()}

            counter = self.collate_results(*results.values())
            label, count = zip(*counter)

        elif isinstance(dataset, Dataset):
            results = {"all": self.report_split(dataset)}
            counter = results["all"].most_common(20)
            label, count = zip(*counter)
        else:
            raise TypeError(f"Unsupported type {type(dataset)}")

        fig = self.plot_splits(label=label, count=count)

        # log results
        self.save_as_html(fig, "top20_match_triggers.html")
