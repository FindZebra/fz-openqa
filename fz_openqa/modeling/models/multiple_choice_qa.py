import re
from typing import Any
from typing import List
from typing import Optional
from typing import Union

from datasets import Split
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from transformers import AdamW
from transformers import BertPreTrainedModel
from transformers import PreTrainedTokenizerFast

from fz_openqa.datamodules.corpus_dm import CorpusDataModule
from fz_openqa.modeling.evaluators.base import BaseEvaluator
from fz_openqa.modeling.models.base import BaseModel
from fz_openqa.modeling.models.multiple_choice_qa_reader import (
    MultipleChoiceQAReader,
)
from fz_openqa.modeling.models.qa_retriever import QaRetriever
from fz_openqa.utils.datastruct import add_prefix
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.datastruct import contains_prefix
from fz_openqa.utils.datastruct import filter_prefix
from fz_openqa.utils.functional import only_trainable


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
        "answer.input_ids",
        "answer.attention_mask",
    ]
    # metrics that will be display in the progress bar
    _prog_bar_metrics = [
        "validation/reader/Accuracy",
        "validation/reader/relevance-Accuracy",
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
        evaluator: Union[BaseEvaluator, DictConfig],
        corpus: Optional[Union[CorpusDataModule, DictConfig]] = None,
        end_to_end_evaluation: bool = False,
        **kwargs,
    ):
        super().__init__(
            **kwargs, evaluator=evaluator, tokenizer=tokenizer, bert=bert
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
        self.end_to_end_evaluation = end_to_end_evaluation
        if self.end_to_end_evaluation:
            self._prog_bar_metrics += ["validation/end2end/Accuracy"]
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

        # supervised step (retriever+reader)
        output = self._supervised_step(batch, batch_idx, dataloader_idx, split)

        # end-to-end step: retrieve document and use the reader
        if split != Split.TRAIN and self.end_to_end_evaluation:
            assert self.evaluator is not None
            output.update(**self.evaluator(self, batch, split=split))

        return output

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
        """
        Compute the final loss and update the metrics for
        1. the retriever
        2. the reader (if available)
        3. the end2end model (if available)
        """

        is_retriever_data_available = contains_prefix("retriever/", output)
        is_end2end_data_available = contains_prefix("end2end/", output)
        kwargs = {"log_data": False, "split": split}

        # process reader data, without logging
        reader_output = self._step_end_reader(output, **kwargs)

        # process retriever data, without logging
        if is_retriever_data_available:
            retriever_output = self._step_end_retriever(output, **kwargs)
        else:
            retriever_output = {}

        # process the end2end data, without logging
        if is_end2end_data_available:
            end2end_output = self._step_end_end2end(output, **kwargs)
        else:
            end2end_output = {}

        # merge and compute the main loss
        output = {**reader_output, **retriever_output, **end2end_output}
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

    def _step_end_end2end(self, output, split: Split, **kwargs):
        end2end_output = self.evaluator.forward_end(output, split)
        end2end_output = add_prefix(end2end_output, "end2end/")
        return end2end_output

    def _epoch_end(self, outputs: List[Any], split: Split, log_data=True):
        kwargs = {"log_data": False, "split": split}
        reader_data = self.reader._epoch_end(outputs, **kwargs)
        if log_data:
            self.log_data(reader_data, prefix=f"{split}/reader/")
        retriever_data = self.retriever._epoch_end(outputs, **kwargs)
        if log_data:
            self.log_data(retriever_data, prefix=f"{split}/retriever/")

        if self.evaluator is not None:
            end2end_data = self.evaluator.compute_metrics(split=split)
            self.evaluator.reset_metrics(split=split)
            if log_data:
                self.log_data(end2end_data, prefix=f"{split}/end2end/")

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
