from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import torch
from datasets import Split
from datasets.search import BatchedNearestExamplesResults
from pytorch_lightning.utilities import move_data_to_device
from torch import nn
from torch import Tensor
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy

from ...datamodules.pipes.nesting import nested_list
from .base import Module
from .metrics import SplitMetrics
from fz_openqa.datamodules.__old.corpus_dm import CorpusDataModule
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.datastruct import filter_prefix
from fz_openqa.utils.datastruct import infer_device_from_batch


class EndToEndMultipleChoiceQaMaximumLikelihood(Module):
    """
    End-to-end evaluation of an OpenQA model (retriever+reader) based on the reader accuracy.
        * Retrieve k documents from the corpus using the retriever model
        * Compute the relevance score for the k-documents using the reader model
        * Predict the answer based on the selected document
    """

    _required_eval_feature_names = [
        "answer.target",
        "question.input_ids",
        "question.attention_mask",
        "answer.input_ids",
        "answer.attention_mask",
    ]

    def __init__(self, n_documents: int, **kwargs):
        super().__init__(**kwargs)
        self.n_documents = n_documents

    def _init_metrics(self, prefix: str = ""):
        """Initialize a Metric for each split=train/validation/test
        fir both the answering model and the selection model"""
        metric_kwargs = {"compute_on_step": False, "dist_sync_on_step": True}

        def init_answer_metric():
            return MetricCollection([Accuracy(**metric_kwargs)], prefix=prefix)

        self.answer_metrics = SplitMetrics(init_answer_metric)

    def step(self, model: nn.Module, batch: Batch, split: Split, **kwargs: Any) -> Batch:
        """
        Compute the forward pass for the question and the documents.

        The input data is assumed to be of shape:
        batch = {
        'question.input_ids': [batch_size, L_q],
        'document.input_ids': [batch_size, n_docs, L_q]
        """
        # check features, check that the first document of each question is positive
        # and flatten the documents
        self._check_batch_type(batch)
        self.check_feature_names(batch)
        assert hasattr(model, "retriever"), "A retriever model must be provided"
        assert hasattr(model, "reader"), "A reader model must be provided"

        device = infer_device_from_batch(batch)

        # query the corpus
        query_encoding = model.retriever(batch, None, None, model_key="question")

        # retriever k documents from the corpus given the query
        retrieved_batch, effective_n_docs = self.retrieve_documents(
            model.retriever.corpus_dataset,
            query_encoding,
            n_docs=self.n_documents,
        )

        reader_data = []
        for idx in range(effective_n_docs):
            # index the k-th document for each batch element and
            # move the batch to device
            retrieved_batch_k = {
                k: v[:, idx] if isinstance(v, Tensor) else [vv[idx] for vv in v]
                for k, v in retrieved_batch.items()
            }
            retrieved_batch_k = move_data_to_device(retrieved_batch_k, device)

            # create a batch with only one document
            batch_k = {k: v for k, v in batch.items() if "document." not in k}
            [batch_k.pop(k) for k in list(batch_k.keys()) if "document." in k]
            batch_k.update(**retrieved_batch_k)

            # forward pass for the reader model
            reader_data += [self._reader_forward(model.reader, batch_k, split)]

        # gather outputs
        reader_data = {
            key: torch.cat([r[key] for r in reader_data], dim=1)
            for key in list(reader_data[0].keys())
        }

        # select the answering logis corresponding to the argmax of the relevance score
        reader_data["answer_logits"] = argmax_select(
            reader_data["answer_logits"], key=reader_data["relevance_logits"]
        )
        reader_data["answer_targets"] = batch["answer.target"]
        return reader_data

    @staticmethod
    def _reader_forward(reader, batch_k: Batch, split: Split) -> Batch:
        kwargs = {"log_data": False, "split": split}
        answer_logits, selection_logits = reader.step(batch_k, **kwargs)
        reader_data_k = {
            "answer_logits": answer_logits,
            "relevance_logits": selection_logits,
            "document.r_rank": batch_k["document.r_rank"].unsqueeze(1),
        }
        return reader_data_k

    @staticmethod
    def retrieve_documents(
        corpus: CorpusDataModule, query: Tensor, n_docs: int
    ) -> Tuple[Batch, int]:
        """
        Retrieve `n_documents` from the corpus object given the `query`.
        """

        batch_size = query.shape[0]
        retrieved_docs: BatchedNearestExamplesResults = corpus.query_batch(query, k=n_docs)
        n_retrieved_docs = len(list(retrieved_docs.total_examples[0].values())[0])

        # create a list of retrieved documents such as:
        # [x[bs_idx=0, r_rank=0], x[bs_idx=0, r_rank=1]], ..., x[bs_idx=1, r_rank=0], ...]
        # NB: r_rank corresponds to the rank of the retrieved doc
        retrieved = retrieved_docs.total_examples
        [r.update({"document.r_rank": -1 + 0 * r["document.idx"]}) for r in retrieved]
        retrieved_batch = [
            {k: idx if k == "document.r_rank" else v[idx] for k, v in d.items()}
            for d in retrieved
            for idx in range(n_retrieved_docs)
        ]

        # collate retrieved documents
        retrieved_batch = corpus.collate_fn(retrieved_batch)

        # reshape all as [batch_size, n_docs, *]
        retrieved_batch = {
            k: v.view(batch_size, n_retrieved_docs, *v.shape[1:])
            if isinstance(v, Tensor)
            else nested_list(v, stride=n_retrieved_docs)
            for k, v in retrieved_batch.items()
        }
        return retrieved_batch, n_retrieved_docs

    def step_end(self, output: Batch, split: Split) -> Any:
        """Apply a post-processing step to the forward method.
        The output is the output of the forward method.

        This method is called after the `output` has been gathered
        from each device. This method must aggregate the loss across
        devices.

        torchmetrics update() calls should be placed here.
        The output must at least contains the `loss` key.
        """
        output = filter_prefix(output, "end2end/")

        for k in ["loss", "relevance_loss", "answer_loss"]:
            y = output.get(k, None)
            if y is not None:
                output[k] = y.mean()

        self.update_metrics(output, split)

        for k in [
            "answer_logits",
            "answer_targets",
            "relevance_logits",
            "relevance_targets",
        ]:
            output.pop(k, None)
        return output

    def update_metrics(self, output: Batch, split: Split) -> None:
        """update the metrics of the given split."""
        answer_logits, answer_targets = (
            output.get(k, None) for k in ("answer_logits", "answer_targets")
        )
        self.answer_metrics.update(split, answer_logits, answer_targets)

    def reset_metrics(self, split: Optional[Split] = None) -> None:
        """
        Reset the metrics corresponding to `split` if provided, else
        reset all the metrics.
        """
        self.answer_metrics.reset(split)

    def compute_metrics(self, split: Optional[Split] = None) -> Batch:
        """
        Compute the metrics for the given `split` else compute the metrics for all splits.
        The metrics are return after computation.
        """
        return self.answer_metrics.compute(split)


def argmax_select(inputs: Tensor, *, key: Tensor) -> Dict[str, Tensor]:
    """
    Index all the tensor in the input based on the armax of the key
    """
    batch_size = key.shape[0]
    arg_max = key.argmax(dim=1)

    # index the reader data using the max `key` position
    def reshape_index(index, v):
        return index.view(batch_size, *(1 for _ in v.shape[1:])).expand(-1, 1, *v.shape[2:])

    return inputs.gather(index=reshape_index(arg_max, inputs), dim=1).squeeze(1)
