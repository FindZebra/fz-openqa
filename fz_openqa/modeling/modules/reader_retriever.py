from __future__ import annotations

import os
from collections import defaultdict
from functools import singledispatch
from typing import Any
from typing import Callable
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

from functools import wraps


def pprint_input_output(silent: bool = False, prefix: str = ""):
    def pprint_input_output_(fn):
        @wraps(fn)
        def wrapper(self, batch: Batch, *args, **kwargs):
            pprint_batch(batch, f"{prefix}{fn.__name__}::input (kwargs={kwargs})", silent=silent)
            output: Batch = fn(self, batch, *args, **kwargs)
            pprint_batch(output, f"{prefix}{fn.__name__}::output", silent=silent)
            return output

        return wrapper

    return pprint_input_output_


def _add_prefix(prefix: str, x: Dict) -> Dict:
    return {prefix + k: v for k, v in x.items()}


def _with_prefix(prefix: str, x: Dict) -> Dict:
    return {k[len(prefix) :]: v for k, v in x.items() if k.startswith(prefix)}


def process_by_chunk(fn: Callable, batch: Batch, *, chunksize: Optional[int], **kwargs) -> Batch:
    """Call `fn` on a batch of `torch.Tensor` by chunk `chunksize`."""

    if chunksize is None:
        return fn(batch, **kwargs)
    else:
        batch_size = next(iter(batch.values())).shape[0]
        outputs = defaultdict(list)
        for i in range(0, batch_size, chunksize):
            chunk = {k: v[i : i + chunksize] for k, v in batch.items()}
            out = fn(chunk, **kwargs)
            for k, v in out.items():
                outputs[k].append(v)
        return {k: torch.cat(v, dim=0) for k, v in outputs.items()}


def process_by_chunk_wrapper(fn):
    @wraps(fn)
    def wrapper(batch: Batch, chunksize: Optional[int] = None, **kwargs):
        output: Batch = process_by_chunk(fn, batch, chunksize=chunksize, **kwargs)
        return output

    return wrapper


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
        eval_chunksize: Optional[int] = None,
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
        self.eval_chunksize = eval_chunksize

        # register the heads
        self.reader: Optional[PreTrainedModel] = maybe_instantiate(reader)
        rich.print(f"Reader: {type(self.reader)}")

        # TODO: handle pretrained VOD (remove tanh)
        # TODO: check that special tokens are encoded (CLS, SEP, PAD)
        # TODO: check that EOS token is appended
        # TODO: output `token_type_ids` for document, question and answer
        self.retriever: Optional[PreTrainedModel] = maybe_instantiate(retriever)

        rich.print(f"Retriever: {type(self.retriever)}")

        # init the estimator
        self.gradients: Gradients = maybe_instantiate(gradients)

    @pprint_input_output(silent=not VERBOSE_MODEL, prefix="ReaderRetriever::")
    def _forward(
        self,
        batch: Batch,
        skip_reader: bool = True,
        skip_retriever: bool = False,
        strict: bool = False,
        **kwargs,
    ) -> Batch:
        process_kwargs = {
            "strict": strict,
            "chunksize": None if self.training else self.eval_chunksize,
        }
        models_outputs = {}
        # process the batch with the reader
        if not skip_reader:
            reader_output = self._process_fields_with_transformer(
                batch,
                model=self.reader,
                fields=self.reader_fields,
                **process_kwargs,
            )
            models_outputs.update(reader_output)

        # process the batch with the retriever
        if not skip_retriever:
            retriever_output = self._process_fields_with_transformer(
                batch,
                model=self.retriever,
                fields=self.retriever_fields,
                **process_kwargs,
            )
            models_outputs.update(retriever_output)

        return models_outputs

    @staticmethod
    def _process_fields_with_transformer(
        batch: Batch,
        *,
        model: PreTrainedModel,
        fields: List[str],
        chunksize: Optional[int] = None,
        strict: bool = True,
    ) -> Batch:
        keys = ["input_ids", "attention_mask", "token_type_ids"]
        output = {}
        for field in fields:
            in_keys = [f"{field}.{k}" for k in keys]
            in_keys = set(in_keys).intersection(batch.keys())
            if len(in_keys) == 0:
                if not strict:
                    continue
                raise ValueError(
                    f"Missing keys {set(in_keys) - set(batch.keys())} "
                    f"in batch for field {field}. "
                    f"Found keys: {batch.keys()}"
                )

            # get batch subset and flatten inputs
            batch_field = {k.replace(f"{field}.", ""): batch[k] for k in in_keys}
            ref_shape = batch_field["input_ids"].shape[:-1]
            batch_field = {k: v.view(-1, *v.shape[-1:]) for k, v in batch_field.items()}

            # forward the model and format the output
            output_field = ReaderRetriever._process_with_transformer(
                batch_field, model=model, chunksize=chunksize
            )

            # un-flatten the output
            for k, v in output_field.items():
                output[f"{field}.{k}"] = v.view(*ref_shape, *v.shape[1:])

        return output

    @staticmethod
    @process_by_chunk_wrapper
    def _process_with_transformer(batch: Batch, *, model: PreTrainedModel) -> Batch:
        output_field = model(**batch)
        output_field = format_transformer_output(output_field, batch=batch)
        return output_field

    @pprint_input_output(silent=not VERBOSE_MODEL, prefix="ReaderRetriever::")
    def _step(self, batch: Batch, silent=not VERBOSE_MODEL, **kwargs: Any) -> Batch:
        # validate the batch
        inputs = ReaderRetrieverInputs(**batch)

        # process the batch with the models
        models_outputs = self._forward(
            inputs.flatten(),
            skip_reader=self.skip_reader,
            skip_retriever=self.skip_retriever,
            strict=True,
            **kwargs,
        )

        # compute the gradients given the outputs from the models
        gradient_step_output = self.gradients.step(
            {
                **models_outputs,
                **batch,
                **kwargs,
            }
        )

        # append prefixes to the outputs for bookkeeping
        step_output = {
            **_add_prefix(STATS_PREFIX, inputs.stats()),
            **_add_prefix(GRAD_PREFIX, gradient_step_output.dict()),
        }
        return step_output

    @pprint_input_output(silent=not VERBOSE_MODEL, prefix="ReaderRetriever::")
    def _reduce_step_output(self, step_output: Batch) -> Batch:
        # retrieve the stats
        output = _with_prefix(STATS_PREFIX, step_output)

        # compute the gradients (step_end)
        output_grad: GradientsOutput = self.gradients.step_end(
            _with_prefix(GRAD_PREFIX, step_output)
        )
        output.update(output_grad.dict())

        # preprare the final output
        final_output = {}
        for k, v in output.items():
            # detach all grads, except the loss
            if k != "loss" and isinstance(v, torch.Tensor):
                v = v.detach()

            # average loss terms, except the terms transmitted to the metrics
            if isinstance(v, torch.Tensor) and not is_feature_name(k):
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

    # get the tensors
    labels = batch["input_ids"]
    mask = batch["attention_mask"]
    token_type_ids = batch.get("token_type_ids", None)

    # Shift so that tokens < n predict n
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_mask = mask[..., 1:].contiguous()
    if token_type_ids is not None:
        shift_token_type_ids = token_type_ids[..., 1:].contiguous()
    else:
        shift_token_type_ids = None

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
