from typing import Any
from typing import Dict
from typing import Optional

from datasets import Dataset

from fz_openqa.datamodules.__old.corpus_dm import CorpusDataModule
from fz_openqa.datamodules.pipes import Batchify
from fz_openqa.datamodules.pipes import DeBatchify
from fz_openqa.datamodules.pipes import RelevanceClassifier


class CorpusSampler:
    """
    Sample the Corpus given a dataset example. This overrides the
    dataset indexing through __getitem__

    This allows adding the documents for each question in parallel (num_workers>0),
    and before the call of collate_fn. This replaces the old approach in collate_fn:
    ```
    batch = self.collate_pipe(examples)
    if self.n_documents > 0 and self.corpus.dataset is not None:
        batch = self.add_documents_to_batch(batch)
    ```

    """

    def __init__(
        self,
        dataset: Dataset,
        corpus: CorpusDataModule,
        n_documents: int,
        relevance_classifier: Optional[RelevanceClassifier] = None,
    ):
        self.dataset = dataset
        self.corpus = corpus
        self.n_documents = n_documents
        self.relevance_classifier = relevance_classifier

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item) -> Dict[str, Any]:
        """get one item from the dataset and update with the retrieved documents.
        The output if of the shape:
        ```
        {
        question.input_ids: shape=[L_eg]
        document.input_ids: list of n_documents values, each of shape=[L_doc_i]
        }
        ```
        """

        # query the original dataset
        example = self.dataset[item]

        # convert the example as a batch (Pipeline compatibility)
        example = Batchify()(example)

        # query the corpus and add the documents to the example
        documents = self.corpus.search_index(
            query=example, k=self.n_documents, simple_collate=True
        )
        example.update(**documents)

        # compute the relevance
        if self.relevance_classifier is not None:
            example = self.relevance_classifier(example)

        # return as one example
        return DeBatchify()(example)
