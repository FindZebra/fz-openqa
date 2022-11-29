from __future__ import annotations

import os
from functools import singledispatch
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import rich
import torch
from datasets import Split
from omegaconf import DictConfig
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from warp_pipes import Batch
from warp_pipes import pprint_batch

from fz_openqa.modeling.datastruct import GRAD_PREFIX
from fz_openqa.modeling.datastruct import METRIC_PREFIX
from fz_openqa.modeling.datastruct import ReaderRetrieverInputs
from fz_openqa.modeling.datastruct import STATS_PREFIX
from fz_openqa.modeling.gradients import Gradients
from fz_openqa.modeling.gradients import RenyiGradients
from fz_openqa.modeling.gradients.base import GradientsOutput
from fz_openqa.modeling.modules.base import is_feature_name
from fz_openqa.modeling.modules.base import Module
from fz_openqa.utils import maybe_instantiate

VERBOSE_MODEL = bool(os.environ.get("VERBOSE_MODEL", False))


def _add_prefix(prefix: str, x: Dict) -> Dict:
    return {prefix + k: v for k, v in x.items()}


def _with_prefix(prefix: str, x: Dict) -> Dict:
    return {k[len(prefix) :]: v for k, v in x.items() if k.startswith(prefix)}


class ReaderRetriever(Module):
    """
    A reader-retriever model for OpenQA.
    """

    _required_feature_names = []

    _required_eval_feature_names = [
        "question.input_ids",
        "question.attention_mask",
        "document.input_ids",
        "document.attention_mask",
    ]

    # prefix for the logged metrics
    task_id: Optional[str] = None

    # metrics to display in the progress bar
    pbar_metrics = [
        "train/reader/logp",
        "validation/reader/logp",
        "train/reader/Accuracy",
        "validation/reader/Accuracy",
        "train/retrieval/MRR",
        "validation/retrieval/MRR",
    ]

    def __init__(
        self,
        *args,
        reader: None | PreTrainedModel | DictConfig = None,
        retriever: None | PreTrainedModel | DictConfig = None,
        retriever_fields: str | List[str] = None,
        reader_fields: str = "lm",
        skip_retriever: bool = False,
        skip_reader: bool = False,
        gradients: Gradients | DictConfig = None,
        **kwargs,
    ):
        if gradients is None:
            gradients = RenyiGradients()

        super().__init__(*args, **kwargs)
        if isinstance(reader_fields, str):
            reader_fields = [reader_fields]
        self.reader_fields = reader_fields
        self.skip_reader = skip_reader
        if retriever_fields is None:
            retriever_fields = ["document", "question"]
        if isinstance(retriever_fields, str):
            retriever_fields = [retriever_fields]
        self.retriever_fields = retriever_fields
        self.skip_retriever = skip_retriever

        # register the heads
        self.reader: Optional[PreTrainedModel] = maybe_instantiate(reader)
        rich.print(f"Reader: {type(self.reader)}")

        # TODO: handle pretrained VOD (remove tanh)
        # TODO: check that special tokens are encoded (CLS, SEP, PAD)
        # TODO: check that EOS token is appended
        # TODO: implement `use token_type_ids` as mask
        self.retriever: Optional[PreTrainedModel] = maybe_instantiate(retriever)

        rich.print(f"Retriever: {type(self.retriever)}")

        # init the estimator
        self.gradients: Gradients = maybe_instantiate(gradients)

    def _forward(
        self,
        batch: Batch,
        skip_reader: bool = True,
        skip_retriever: bool = False,
        strict: bool = False,
        **kwargs,
    ) -> Batch:
        models_outputs = {}
        # process the batch with the reader
        if not skip_reader:
            reader_output = self._process_with_transformer(
                self.reader, batch, self.reader_fields, strict=strict
            )
            models_outputs.update(reader_output)

        # process the batch with the retriever
        if not skip_retriever:
            retriever_output = self._process_with_transformer(
                self.retriever, batch, self.retriever_fields, strict=strict
            )
            models_outputs.update(retriever_output)

        return models_outputs

    @staticmethod
    def _process_with_transformer(
        model: PreTrainedModel,
        batch: Batch,
        fields: List[str],
        strict: bool = True,
    ) -> Batch:
        keys = ["input_ids", "attention_mask", "token_type_ids"]
        output = {}
        for field in fields:
            in_keys = [f"{field}.{k}" for k in keys]
            if not set(in_keys).issubset(batch.keys()):
                if not strict:
                    continue
                raise ValueError(
                    f"Missing keys {set(in_keys) - set(batch.keys())} "
                    f"in batch for field {field}. "
                    f"Found keys: {batch.keys()}"
                )

            batch_field = {k.replace(f"{field}.", ""): batch[k] for k in in_keys}
            ref_shape = batch_field["input_ids"].shape[:-1]
            batch_field = {k: v.view(-1, *v.shape[-1:]) for k, v in batch_field.items()}

            pprint_batch(
                batch_field,
                f"process_with_transformer::batch_field::{field} (model={type(model)})",
                silent=not VERBOSE_MODEL,
            )
            output_field = model(**batch_field)
            output_field = format_transformer_output(output_field, batch=batch_field)
            pprint_batch(
                output_field,
                f"process_with_transformer::output_field::{field}",
                silent=not VERBOSE_MODEL,
            )
            for k, v in output_field.items():
                output[f"{field}.{k}"] = v.view(*ref_shape, *v.shape[1:])

        return output

    def _step(self, batch: Batch, silent=not VERBOSE_MODEL, **kwargs: Any) -> Batch:
        """
        Compute the forward pass for the question and the documents.
        """

        inputs = ReaderRetrieverInputs(**batch)

        pprint_batch(batch, "ReaderRetriever::_step::batch", silent=silent)
        models_outputs = self._forward(
            inputs.flatten(),
            skip_reader=self.skip_reader,
            skip_retriever=self.skip_retriever,
            strict=True,
            **kwargs,
        )
        pprint_batch(models_outputs, "ReaderRetriever::_step::models_outputs", silent=silent)

        # compute the gradients
        gradient_step_output = self.gradients.step(
            {
                **models_outputs,
                **batch,
                **kwargs,
            }
        )

        # prepare the output
        step_output = {
            **_add_prefix(STATS_PREFIX, inputs.stats()),
            **_add_prefix(GRAD_PREFIX, gradient_step_output.dict()),
        }
        pprint_batch(step_output, "ReaderRetriever::_step::step_output", silent=silent)
        return step_output

    def _reduce_step_output(self, step_output: Batch) -> Batch:
        """
        Gather scores and compute the gradients
        """
        pprint_batch(
            step_output,
            "ReaderRetriever::_step_output::step_output",
            silent=not VERBOSE_MODEL,
        )

        # retrieve the stats
        output = _with_prefix(STATS_PREFIX, step_output)

        # compute the gradients (step_end)
        output_grad: GradientsOutput = self.gradients.step_end(
            _with_prefix(GRAD_PREFIX, step_output)
        )
        output.update(output_grad.dict())

        pprint_batch(
            output,
            "ReaderRetriever::_reduce_step_output::output",
            silent=not VERBOSE_MODEL,
        )

        # preprare the final output
        final_output = {}
        for k, v in output.items():
            # detach all grads, except the loss
            if k != "loss" and isinstance(v, torch.Tensor):
                v = v.detach()

            # average loss terms, except the terms transmitted to the metrics
            if k != "loss" and not is_feature_name(k):
                if isinstance(v, torch.Tensor):
                    v = v.float().mean()

            # format name and store
            k = k.replace(".", "/")
            final_output[k] = v

        pprint_batch(final_output, "reduce_step_output::final_output", silent=not VERBOSE_MODEL)

        return final_output

    @torch.no_grad()
    def update_metrics(self, output: Batch, split: Split) -> None:
        """update the metrics of the given split."""

        # log the reader accuracy
        reader_logits = output.get(f"{METRIC_PREFIX}reader.logits", None)
        reader_targets = output.get(f"{METRIC_PREFIX}reader.targets", None)
        if reader_targets is not None:
            if reader_logits is not None:
                self.reader_metrics.update(split, reader_logits, reader_targets)

        # log the retriever accuracy
        retriever_logits = output.get(f"{METRIC_PREFIX}retriever.logits", None)
        retriever_targets = output.get(f"{METRIC_PREFIX}retriever.targets", None)
        if retriever_logits is not None and retriever_logits.numel() > 0:
            if retriever_targets is not None:
                self.retriever_metrics.update(split, retriever_logits, retriever_targets)

    def reset_metrics(self, split: Optional[Split] = None) -> None:
        """
        Reset the metrics corresponding to `split` if provided, else
        reset all the metrics.
        """
        self.reader_metrics.reset(split)
        self.retriever_metrics.reset(split)

    def compute_metrics(self, split: Optional[Split] = None) -> Batch:
        """
        Compute the metrics for the given `split` else compute the metrics for all splits.
        The metrics are return after computation.
        """
        return {
            **self.reader_metrics.compute(split),
            **self.retriever_metrics.compute(split),
        }

    def step_end(
        self,
        output: Batch,
        split: Optional[Split],
        update_metrics: bool = True,
        filter_features: bool = True,
    ) -> Any:
        """
        Force returning features during validation: required by `LogPredictionsCallback`
        """
        if split is not None and not split == Split.TRAIN:
            filter_features = False

        return super().step_end(
            output, split, filter_features=filter_features, update_metrics=update_metrics
        )

    def _init_metrics(self, prefix: str):
        """Initialize the metrics for each split."""
        self.reader_metrics = self._get_base_metrics(prefix=f"{prefix}reader/")

        self.retriever_metrics = self._get_base_metrics(
            prefix=f"{prefix}retrieval/",
            topk=[1, 3, 5, 10, 20, 50, 100],
            retrieval=True,
            allowed_splits=[Split.VALIDATION, Split.TEST],
        )


@singledispatch
def format_transformer_output(output: Any, *, batch: Batch) -> Batch:
    """
    Format the output of the transformer to a batch.
    """
    raise TypeError(f"Cannot handle output of type {type(output)}")


@format_transformer_output.register(CausalLMOutputWithCrossAttentions)
def _(output: CausalLMOutputWithCrossAttentions, *, batch: Batch) -> Batch:
    try:
        lm_logits = output.logits
    except AttributeError as exc:
        raise ValueError(
            f"The output of the transformer must have a `logits` attribute. "
            f"Found {dict(output).keys()}"
        ) from exc
    labels = batch["input_ids"]
    mask = batch["attention_mask"]
    token_type_ids = batch.get("token_type_ids", None)
    # Shift so that tokens < n predict n
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_mask = mask[..., 1:].contiguous()
    if token_type_ids is not None:
        shift_token_type_ids = token_type_ids[..., 1:].contiguous()
    # Flatten the tokens and compute log `p(x[1:T])`
    loss_fct = CrossEntropyLoss(reduction="none")
    logp_tokens = -loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    logp_tokens = logp_tokens.view_as(shift_mask)
    logp_tokens = logp_tokens * shift_mask

    # compute the loss terms
    output = {"logp": logp_tokens.sum(dim=-1)}
    if token_type_ids is not None:
        output.update(
            {
                "logp_q": (logp_tokens * (shift_token_type_ids == 0).float()).sum(dim=1),
                "logp_a": (logp_tokens * (shift_token_type_ids == 1).float()).sum(dim=1),
            }
        )

    return output


@format_transformer_output.register(BaseModelOutputWithPoolingAndCrossAttentions)
def _(output: BaseModelOutputWithPoolingAndCrossAttentions, *, batch: Batch) -> Batch:
    return {
        "pooler_output": output.pooler_output,
    }
