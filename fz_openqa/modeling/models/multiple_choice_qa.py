import re
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import torch
from datasets import Split
from datasets.search import BatchedNearestExamplesResults
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.utilities import move_data_to_device
from torch import nn
from torch import Tensor
from transformers import AdamW
from transformers import BertPreTrainedModel
from transformers import PreTrainedTokenizerFast

from fz_openqa.datamodules.corpus_dm import CorpusDataModule
from fz_openqa.modeling.models.base import BaseModel
from fz_openqa.modeling.models.multiple_choice_qa_reader import (
    MultipleChoiceQAReader,
)
from fz_openqa.modeling.models.qa_retriever import QaRetriever
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.datastruct import infer_device_from_batch
from fz_openqa.utils.functional import only_trainable


def add_prefix(d: Dict[str, Any], prefix: str):
    return {f"{prefix}{k}": v for k, v in d.items()}


def filter_prefix(d: Dict[str, Any], prefix: str):
    return {k.replace(prefix, ""): v for k, v in d.items() if prefix in k}


class MultipleChoiceQA(BaseModel):
    """
    An end-to-end multiple choice openQA model with:
    * a dense retriever
    * a multiple-choice reader
    """

    _required_feature_names = [
        "question.input_ids",
        "question.attention_mask",
        "document.input_ids",
        "document.attention_mask",
        "answer_choices.input_ids",
        "answer_choices.attention_mask",
    ]
    # metrics that will be display in the progress bar
    _prog_bar_metrics = [
        "train/loss",
        "validation/loss",
        "validation/reader/Accuracy",
        "validation/retriever/Accuracy",
        "validation/retriever/top5_Accuracy",
    ]
    # prefix for the logged metrics
    _model_log_prefix = ""

    def __init__(
        self,
        *,
        tokenizer: PreTrainedTokenizerFast,
        bert: Union[BertPreTrainedModel, DictConfig],
        reader: Union[DictConfig, MultipleChoiceQAReader],
        retriever: Union[DictConfig, QaRetriever],
        corpus: Optional[Union[CorpusDataModule, DictConfig]] = None,
        eval_using_corpus: bool = False,
        eval_top_k: int = 2,
        **kwargs,
    ):
        super().__init__(
            **kwargs, evaluator=None, tokenizer=tokenizer, bert=bert
        )

        # instantiate the reader
        self.reader: MultipleChoiceQAReader = instantiate(
            reader, bert=self.bert, tokenizer=tokenizer
        )

        # instantiate the retriever
        self.retriever: QaRetriever = instantiate(
            retriever, bert=self.bert, tokenizer=tokenizer, corpus=corpus
        )

        # end-to-end evaluation
        self.eval_top_k = eval_top_k
        self.eval_using_corpus = eval_using_corpus
        if self.eval_using_corpus:
            assert (
                corpus is not None
            ), "A corpus object must be provided in order to evaluate using the corpus."

    def _step(
        self,
        batch: Any,
        batch_idx: int,
        dataloader_idx: Optional[int],
        *,
        split: Split,
        **kwargs,
    ) -> Batch:

        if batch.pop("_mode_", None) == "indexing":
            return self.retriever.predict_step(
                batch, batch_idx, dataloader_idx
            )

        if split == Split.TRAIN or not self.eval_using_corpus:
            return self._supervised_step(
                batch, batch_idx, dataloader_idx, split
            )
        else:
            return self._end_to_end_step(
                batch, batch_idx, dataloader_idx, split
            )

    def _end_to_end_step(self, batch, batch_idx, dataloader_idx, split):
        """
        Perform an end-to-end evaluation step:
         1. query the corpus using the retriever
         2. evaluate the reader using the retriever document

        todo: wrap this in an evaluator class
        todo: batch the for loop with `train_top_k` documents
        """
        device = infer_device_from_batch(batch)

        # query the corpus
        query_encoding = self.retriever(
            batch, batch_idx, dataloader_idx, model_key="question"
        )

        # retriever k documents from the corpus given the query
        retrieved_batch = self.retrieve_documents(
            self.retriever.corpus, query_encoding, n_docs=self.eval_top_k
        )

        reader_data = []
        for idx in range(self.eval_top_k):
            # index and move the batc
            retrieved_batch_k = {
                k: v[:, idx] for k, v in retrieved_batch.items()
            }
            retrieved_batch_k = move_data_to_device(retrieved_batch_k, device)

            # create a batch with only one document
            batch_k = {k: v for k, v in batch.items() if "document." not in k}
            [batch_k.pop(k) for k in list(batch_k.keys()) if "document." in k]
            batch_k.update(**retrieved_batch_k)

            # forward pass for the reader model
            reader_data += [self._reader_forward(batch_k, split)]

        # gather outputs
        # todo: experiment with sum(p_select * answer_logits) instead of argmax
        reader_data = {
            key: torch.cat([r[key] for r in reader_data], dim=1)
            for key in list(reader_data[0].keys())
        }
        reader_data["answer_logits"] = self.argmax_select(
            reader_data["answer_logits"], key=reader_data["select_logits"]
        )
        reader_data["answer_targets"] = batch["answer_idx"]

        # add key prefix and return
        output = add_prefix(reader_data, "reader/")
        return output

    def _reader_forward(self, batch_k: Batch, split: Split) -> Batch:
        kwargs = {"log_data": False, "split": split}
        answer_logits, selection_logits = self.reader.forward(
            batch_k, **kwargs
        )
        reader_data_k = {
            "answer_logits": answer_logits,
            "select_logits": selection_logits,
            "r_rank": batch_k["r_rank"].unsqueeze(1),
        }
        return reader_data_k

    def retrieve_documents(
        self, corpus: CorpusDataModule, query: Tensor, n_docs: int
    ) -> Batch:
        """
        Retrieve `n_documents` from the corpus object given the `query`.
        """

        batch_size = query.shape[0]
        retrieved_docs: BatchedNearestExamplesResults = corpus.query_batch(
            query, k=n_docs
        )
        # create a list of retrieved documents such as:
        # [x[r_rank=0, bs_idx=0], x[r_rank=1, bs_idx=0]], ...x[r_rank=0, bs_idx=1]]
        # NB: r_rank corresponds to the rank of the retrieved doc
        retrieved = retrieved_docs.total_examples
        [r.update({"r_rank": -1 + 0 * r["idx"]}) for r in retrieved]
        retrieved_batch = [
            {k: idx if k == "r_rank" else v[idx] for k, v in d.items()}
            for d in retrieved
            for idx in range(n_docs)
        ]

        # collate retrieved documents
        retrieved_batch = self.retriever.corpus.collate_fn(retrieved_batch)

        # reshape all as [batch_size, n_docs, *]
        retrieved_batch = {
            k: v.view(batch_size, n_docs, *v.shape[1:])
            for k, v in retrieved_batch.items()
        }
        return retrieved_batch

    @staticmethod
    def argmax_select(inputs: Tensor, *, key: Tensor) -> Dict[str, Tensor]:
        """
        Index all the tensor in the input based
        """
        batch_size = key.shape[0]
        arg_max = key.argmax(dim=1)

        # index the reader data using the max `key` position
        def reshape_index(index, v):
            return index.view(batch_size, *(1 for _ in v.shape[1:])).expand(
                -1, 1, *v.shape[2:]
            )

        return inputs.gather(
            index=reshape_index(arg_max, inputs), dim=1
        ).squeeze(1)

    def _supervised_step(self, batch, batch_idx, dataloader_idx, split):
        """Perform an evaluation step using the triplets (q,d,a). The reader using the
        golden document. The retriever is optimized using the DPR loss. The losses of the
        retriever and readers are computed independently from each other."""
        output = {}
        kwargs = {"log_data": False, "split": split}

        # forward pass for the retriever model
        retriever_data = self.retriever._step(
            batch, batch_idx, dataloader_idx, **kwargs
        )
        output.update(add_prefix(retriever_data, "retriever/"))

        # forward pass for the reader model
        reader_data = self.reader._step(
            batch, batch_idx, dataloader_idx, **kwargs
        )
        output.update(add_prefix(reader_data, "reader/"))

        return output

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Any:
        # compute contextualized representations
        mode = batch.pop("mode", None)
        assert (
            mode is not None
        ), f"A `mode` argument must be provided, batch.keys()={list(batch.keys())}"
        if mode == "indexing":
            return self.retriever.predict_step(
                batch, batch_idx, dataloader_idx
            )
        else:
            raise NotImplementedError

    def _step_end(self, output: Batch, split, log_data=True) -> Batch:
        # update the metrics for both the reader and retriever
        is_retriever_data_available = any(
            "retriever/" in k for k in output.keys()
        )
        kwargs = {"log_data": False, "split": split}

        # process reader data, without logging
        reader_output = self._step_end_reader(output, **kwargs)

        # process retriever data, without logging
        if is_retriever_data_available:
            retriever_output = self._step_end_retriever(output, **kwargs)
        else:
            retriever_output = {}

        # merge and compute the main loss
        output = {**reader_output, **retriever_output}
        output["loss"] = output.get("reader/loss", 0) + output.get(
            "retriever/loss", 0
        )

        # log the data for both the reader and the retriever
        if log_data:
            self.log_data(output, prefix=f"{split}/")

        return output

    def _step_end_reader(self, output, **kwargs):
        reader_output = self.reader._step_end(
            filter_prefix(output, "reader/"), **kwargs
        )
        reader_output = add_prefix(reader_output, "reader/")
        return reader_output

    def _step_end_retriever(self, output, **kwargs):
        retriever_output = self.retriever._step_end(
            filter_prefix(output, "retriever/"),
            **kwargs,
        )
        retriever_output = add_prefix(retriever_output, "retriever/")
        return retriever_output

    def _epoch_end(self, outputs: List[Any], split: Split, log_data=True):
        kwargs = {"log_data": False, "split": split}
        reader_data = self.reader._epoch_end(outputs, **kwargs)
        if log_data:
            self.log_data(reader_data, prefix=f"{split}/reader/")
        retriever_data = self.retriever._epoch_end(outputs, **kwargs)
        if log_data:
            self.log_data(retriever_data, prefix=f"{split}/retriever/")

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """

        def filtered_params(model: nn.Module, pattern="^bert."):
            return (
                p
                for k, p in model.named_parameters()
                if not re.findall(pattern, k)
            )

        return AdamW(
            [
                {
                    "params": only_trainable(self.bert.parameters()),
                    "lr": float(self.hparams.lr),
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": only_trainable(filtered_params(self.retriever)),
                    "lr": float(self.retriever.hparams.lr),
                    "weight_decay": self.retriever.hparams.weight_decay,
                },
                {
                    "params": only_trainable(filtered_params(self.reader)),
                    "lr": float(self.reader.hparams.lr),
                    "weight_decay": self.reader.hparams.weight_decay,
                },
            ],
        )
