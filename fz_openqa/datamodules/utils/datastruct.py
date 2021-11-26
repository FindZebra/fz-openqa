from datasets import Dataset
from datasets import DatasetDict

from fz_openqa.datamodules.index import Index


class OpenQaDataset(DatasetDict):
    def __init__(self, *, dataset: DatasetDict, corpus: Dataset, index: Index):
        super(OpenQaDataset, self).__init__(dataset)
        self.corpus = corpus
        self.index = index

    def new(self, *, dataset: DatasetDict) -> "OpenQaDataset":
        return OpenQaDataset(dataset=dataset, corpus=self.corpus, index=self.index)

    def __repr__(self):
        u = f"{self.__class__.__name__}:\n"
        u += f" - dataset={super().__repr__()}\n"
        u += f" - corpus={self.corpus}\n"
        u += f" - index={self.index}\n"
        return u
