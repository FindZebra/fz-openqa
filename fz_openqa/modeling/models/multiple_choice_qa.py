import re
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from datasets import Split
from datasets.search import BatchedNearestExamplesResults
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
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
        # query the corpus
        query_encoding = self.retriever(
            batch, batch_idx, dataloader_idx, model_key="question"
        )
        retrieved_docs: BatchedNearestExamplesResults = (
            self.retriever.corpus.query_batch(query_encoding, k=1)
        )
        retrieved_batch = [
            {k: v[0] for k, v in d.items()}
            for d in retrieved_docs.total_examples
        ]
        retrieved_batch = self.retriever.corpus.collate_fn(retrieved_batch)
        device = next(iter(batch.values())).device
        retrieved_batch = {k: v.to(device) for k, v in retrieved_batch.items()}
        # merge retriever batch with batch
        [batch.pop(k) for k in list(batch.keys()) if "document." in k]
        batch.update({f"document.{k}": v for k, v in retrieved_batch.items()})
        # reader only
        kwargs = {"log_data": False, "split": split}
        # forward pass for the reader model
        reader_data = self.reader._step(
            batch, batch_idx, dataloader_idx, **kwargs
        )
        output = add_prefix(reader_data, "reader/")
        return output

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
        has_retriever_data = any("retriever/" in k for k in output.keys())
        kwargs = {"log_data": False, "split": split}

        # process reader data, without logging
        reader_output = self.reader._step_end(
            filter_prefix(output, "reader/"), **kwargs
        )
        reader_output = add_prefix(reader_output, "reader/")

        # process retriever data, without logging
        if has_retriever_data:
            retriever_output = self.retriever._step_end(
                filter_prefix(output, "retriever/"),
                **kwargs,
            )
            retriever_output = add_prefix(retriever_output, "retriever/")

            # merge and compute the main loss
            output = {**reader_output, **retriever_output}
            output["loss"] = output["reader/loss"] + output["retriever/loss"]

        else:
            output = reader_output
            output["loss"] = output["reader/loss"]

        # log the data for both the reader and the retriever
        if log_data:
            self.log_data(output, prefix=f"{split}/")

        return output

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
