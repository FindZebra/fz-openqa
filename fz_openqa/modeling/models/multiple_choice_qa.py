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
from torch import nn
from torch import Tensor
from transformers import AdamW
from transformers import BertPreTrainedModel
from transformers import PreTrainedTokenizerFast

from ...datamodules.corpus_dm import CorpusDataModule
from ...utils.functional import only_trainable
from .base import BaseModel
from .multiple_choice_qa_reader import MultipleChoiceQAReader
from .qa_retriever import QaRetriever
from fz_openqa.utils.datastruct import Batch


def add_prefix(d: Dict[str, Any], prefix: str):
    return {f"{prefix}{k}": v for k, v in d.items()}


def filter_prefix(d: Dict[str, Any], prefix: str):
    return {k.replace(prefix, ""): v for k, v in d.items() if prefix in k}


class MultipleChoiceQA(BaseModel):
    """A Multiple Choice model """

    _required_infer_feature_names = [
        "question.input_ids",
        "question.attention_mask",
        "question.input_ids",
        "document.attention_mask",
        "question.input_ids",
        "question.attention_mask",
        "answer_choices.input_ids",
        "answer_choices.attention_mask",
    ]
    _prog_bar_metrics = [
        "train/loss",
        "validation/loss",
        "validation/reader/Accuracy",
        "validation/retriever/Accuracy",
        "validation/retriever/top5_Accuracy",
    ]  # metrics that will be display in the progress bar

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
        super().__init__(**kwargs, evaluator=None)

        # instantiate the pretrained model
        self.instantiate_bert(bert=bert, tokenizer=tokenizer)

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

        if batch.pop("mode", None) == "indexing":
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

        todo: sample multiple documents from the corpus
        todo: wrap this in an evaluator class
        """
        device = next(iter(batch.values())).device
        batch_size = batch["question.input_ids"].shape[0]

        # query the corpus
        query_encoding = self.retriever(
            batch, batch_idx, dataloader_idx, model_key="question"
        )

        # cast as a batch
        retrieved_docs: BatchedNearestExamplesResults = (
            self.retriever.corpus.query_batch(
                query_encoding, k=self.eval_top_k
            )
        )

        # create a list of retrieved documents such as:
        # [x[r_rank=0, bs_idx=0], x[r_rank=1, bs_idx=0]], ...x[r_rank=0, bs_idx=1]]
        # NB: r_rank corresponds to the rank of the retrieved doc
        retrieved = retrieved_docs.total_examples
        [r.update({"r_rank": -1 + 0 * r["idx"]}) for r in retrieved]
        retrieved_batch = [
            {k: idx if k == "r_rank" else v[idx] for k, v in d.items()}
            for idx in range(self.eval_top_k)
            for d in retrieved
        ]

        retrieved_batch = self.retriever.corpus.collate_fn(retrieved_batch)
        # retrieved_batch = {k: v.to(device) for k, v in retrieved_batch.items()}

        # reshape all as [k, bs, *]
        retrieved_batch = {
            k: v.view(self.eval_top_k, batch_size, *v.shape[1:])
            for k, v in retrieved_batch.items()
        }

        reader_data = []
        for idx in range(self.eval_top_k):
            retrieved_batch_k = {k: v[idx] for k, v in retrieved_batch.items()}
            retrieved_batch_k = {
                k: v.to(device) for k, v in retrieved_batch_k.items()
            }
            batch_k = batch
            # merge retriever batch with batch
            [batch_k.pop(k) for k in list(batch_k.keys()) if "document." in k]
            batch_k.update(
                {f"document.{k}": v for k, v in retrieved_batch_k.items()}
            )

            # forward pass for the reader model
            kwargs = {"log_data": False, "split": split}
            reader_data_k = self.reader._step(
                batch_k, batch_idx, dataloader_idx, **kwargs
            )

            reader_data_k["r_rank"] = retrieved_batch_k["r_rank"]
            reader_data += [reader_data_k]

        # gather outputs
        # temporary hack: select the reader output with the highest logit
        # todo: implement a proper selection model: i.e. eq 5 in DPR (https://arxiv.org/pdf/2004.04906.pdf)
        [
            r.update({"largest_logit": r["logits"].softmax(-1).max(dim=-1)[0]})
            for r in reader_data
        ]
        reader_data = self.argmax_select(reader_data, key="largest_logit")

        # add key prefix and return
        output = add_prefix(reader_data, "reader/")
        return output

    @staticmethod
    def argmax_select(
        inputs: List[Dict[str, Tensor]], *, key
    ) -> Dict[str, Tensor]:
        """
        Given a list of M inputs, each encoded as `Dict[str, Tensor]` where all tensors
        are of the same batch size, this function return one input Dict[str] where for each batch element (independently),
        the output is the input having the largest `key` value
        """

        assert all(
            isinstance(t, Tensor) for r in inputs for t in r.values()
        ), "Inputs must all be tensors"
        batch_size = inputs[0][key].shape[0]
        assert all(
            t.shape[0] == batch_size for r in inputs for t in r.values()
        ), "Tensors must be of the same batch size"
        keys = list(inputs[0].keys())
        assert all(
            set(r.keys()) == set(keys) for r in inputs
        ), "Inputs must all have the same keys"
        assert all(
            r[key].shape == (batch_size,) for r in inputs
        ), "input[key] must all be of shape [batch_size]"

        # concatenate all inputs across dimension 1
        inputs = {
            k: torch.cat([r[k][:, None] for r in inputs], dim=1) for k in keys
        }  # shape [bs, M, *]
        arg_max = inputs[key].argmax(dim=1)

        # index the reader data using the max `key` position
        def reshape_index(index, v):
            return index.view(batch_size, *(1 for _ in v.shape[1:])).expand(
                -1, 1, *v.shape[2:]
            )

        inputs = {
            k: v.gather(index=reshape_index(arg_max, v), dim=1).squeeze(1)
            for k, v in inputs.items()
        }
        return inputs

    def _supervised_step(self, batch, batch_idx, dataloader_idx, split):
        """Perform an evaluation step using the triplets (q,d,a). The reader using the
        golden document. The retriever is optimized using the DPR loss. The losses of the
        retriever and readers are computed independently from each other."""
        output = {}
        kwargs = {"log_data": False, "split": split}
        # forward pass for the reader model
        reader_data = self.reader._step(
            batch, batch_idx, dataloader_idx, **kwargs
        )
        output.update(add_prefix(reader_data, "reader/"))
        # forward pass for the retriever model
        retriever_data = self.retriever._step(
            batch, batch_idx, dataloader_idx, **kwargs
        )
        output.update(add_prefix(retriever_data, "retriever/"))
        return output

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Any:
        # todo: investigate why this is not used (see trick with adding the key `mode` to the batch in _step)
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
