from typing import Union, Dict
import itertools

import plotly.express as px
from collections import Counter
from datasets import Dataset
from datasets import DatasetDict

from ..utils.datastruct import OpenQaDataset
from ..utils.typing import HfDataset
from .base import Analytic
from fz_openqa.datamodules.utils.dataset import get_column_names


class PlotTop20MatchTriggers(Analytic):
    """Count the number of questions matched with positive documents"""
    @staticmethod
    def count_split(dset: Dataset) -> Dict:
        """
        Plot a specific split of the dataset.
        """
        n_triggers = dset["document.match_on"]
        n_triggers = [y[0] for y in n_triggers]
        triggers = list(itertools.chain(*n_triggers))
        print(triggers)
        count_triggers = Counter(triggers)

        return count_triggers

    def __call__(self, dataset: Union[HfDataset, OpenQaDataset], **kwargs):
        """Gather retrieval scores of positive documents and plot distribution."""
        assert "document.match_score" in get_column_names(dataset)
        if isinstance(dataset, DatasetDict):
            results = {split: self.count_split(dset) for split, dset in dataset.items()}

            counter = results['train'] + results['validation'] + results['test']
            counter = counter.most_common(20)

            label, count = zip(*counter)

        elif isinstance(dataset, Dataset):
            results = {"all": self.count_split(dataset)}

            counter = results['all'].most_common(20)

            label, count = zip(*counter)
        else:
            raise TypeError(f"Unsupported type {type(dataset)}")

        # create figure
        fig = px.bar(y=count, x=label, text=count)
        fig.update_traces(textposition='outside', marker_color='rgb(241, 186, 195)')
        fig.update_yaxes(ticklabelposition="inside top", title="Count")
        fig.update_xaxes(title="Triggers")
        # log results
        self.save_as_html(fig, "top20_match_triggers.html")
