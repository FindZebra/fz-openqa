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
from pydantic import BaseModel
from torch import Tensor
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizerFast
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.modeling_outputs import CausalLMOutputWithPast
from warp_pipes import Batch
from warp_pipes import pprint_batch

from fz_openqa.modeling.datastruct import GRAD_PREFIX
from fz_openqa.modeling.datastruct import METRICS_PREFIX
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


class ModelOutputFormatConfig(BaseModel):
    """Configuration for the output format of a model."""

    class Config:
        arbitrary_types_allowed = True

    answer_choices_tokens: Optional[Tensor] = None
    tokenizer: Optional[PreTrainedTokenizerFast] = None


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


def _with_prefix(prefix: str | List[str], x: Dict, strip_prefix: bool = True) -> Dict:
    if isinstance(prefix, str):
        prefix = [prefix]
    output = {}
    for k, v in x.items():
        for p in prefix:
            if str(k).startswith(p):
                if strip_prefix:
                    k = k[len(p) :]
                output[k] = v
                break

    return output


def process_by_chunk(fn: Callable, batch: Batch, *, chunksize: Optional[int], **kwargs) -> Batch:
    """Call `fn` on a batch of `torch.Tensor` by chunk `chunksize`."""

    if chunksize is None:
        return fn(batch, **kwargs)
    else:
        # TODO: scale chunksize with size
        # TODO: allocate memory at first chunk
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

    # TODO: refactor/unify input validation
    _required_feature_names = []
    _required_eval_feature_names = []

    # prefix for the logged metrics
    task_id: Optional[str] = None

    # metrics to display in the progress bar
    pbar_metrics = [
        "train/lm/logp",
        "validation/lm/logp",
        "train/answer/Accuracy",
        "validation/answer/Accuracy",
        "train/document/MRR",
        "validation/document/MRR",
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
        reader_tokenizer: None | PreTrainedTokenizerFast | DictConfig = None,
        infer_multiple_choice_logits: bool = True,
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

        # freeze parameters
        # if self.skip_retriever:
        #     for param in self.retriever.parameters():
        #         param.requires_grad = False
        # if self.skip_reader:
        #     for param in self.reader.parameters():
        #         param.requires_grad = False

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

        # store the answer choices token_ids
        reader_tokenizer = maybe_instantiate(reader_tokenizer)
        answer_tokens = reader_tokenizer(
            ["A", "B", "C", "D"], return_tensors="pt", add_special_tokens=False
        ).input_ids.squeeze(1)
        self.register_buffer("answer_tokens", answer_tokens)
        self.infer_multiple_choice_logits = infer_multiple_choice_logits
        # todo: remove this
        self.tokenizer = reader_tokenizer

    @property
    def model_output_format_config(self) -> ModelOutputFormatConfig:
        answer_choices_tokens = self.answer_tokens if self.infer_multiple_choice_logits else None
        return ModelOutputFormatConfig(
            answer_choices_tokens=answer_choices_tokens,
            tokenizer=self.tokenizer,
        )

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
            "config": self.model_output_format_config,
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
        **kwargs,
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
            batch_field = {
                k.replace(f"{field}.", ""): batch[k] for k in in_keys if batch[k] is not None
            }
            ref_shape = batch_field["input_ids"].shape[:-1]
            batch_field = {k: v.view(-1, *v.shape[-1:]) for k, v in batch_field.items()}

            # forward the model and format the output
            output_field = ReaderRetriever._process_with_transformer(
                batch_field,
                model=model,
                chunksize=chunksize,
                **kwargs,
            )

            # un-flatten the output
            for k, v in output_field.items():
                output[f"{field}.{k}"] = v.view(*ref_shape, *v.shape[1:])

        return output

    @staticmethod
    @process_by_chunk_wrapper
    def _process_with_transformer(
        batch: Batch, *, model: PreTrainedModel, config: ModelOutputFormatConfig
    ) -> Batch:
        output_field = model(**batch)
        output_field = format_transformer_output(output_field, batch=batch, config=config)
        return output_field

    @pprint_input_output(silent=not VERBOSE_MODEL, prefix="ReaderRetriever::")
    def _step(self, batch: Batch, silent=not VERBOSE_MODEL, **kwargs: Any) -> Batch:
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

        # pass the targets to the metrics
        if inputs.answer is not None and inputs.answer.target is not None:
            step_output[f"{METRICS_PREFIX}answer.target"] = inputs.answer.target
        if inputs.document is not None and "target" in inputs.document:
            step_output[f"{METRICS_PREFIX}document.target"] = inputs.document.target

        return step_output

    @pprint_input_output(silent=not VERBOSE_MODEL, prefix="ReaderRetriever::")
    def _reduce_step_output(self, step_output: Batch) -> Batch:
        # retrieve the stats
        output = {
            **_with_prefix(STATS_PREFIX, step_output),
            **_with_prefix(METRICS_PREFIX, step_output, strip_prefix=False),
        }

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
            if not str(k).startswith(METRICS_PREFIX):
                k = k.replace(".", "/")
            final_output[k] = v

        pprint_batch(final_output, "reduce_step_output::final_output", silent=not VERBOSE_MODEL)

        return final_output

    @torch.inference_mode()
    def update_metrics(self, final_output: Batch, split: Split) -> None:
        """update the metrics of the given split."""

        final_output = _with_prefix(METRICS_PREFIX, final_output, strip_prefix=True)
        pprint_batch(final_output, "update_metrics::output", silent=not VERBOSE_MODEL)

        # log the reader accuracy
        reader_logits = final_output.get("answer.logits", None)
        reader_targets = final_output.get("answer.target", None)
        if reader_targets is not None:
            if reader_logits is not None:
                self.reader_metrics.update(split, reader_logits, reader_targets)

        # log the retriever accuracy
        retriever_logits = final_output.get("document.logits", None)
        retriever_targets = final_output.get("document.target", None)
        if retriever_logits is not None and retriever_logits.numel() > 0:
            if retriever_targets is not None:
                self.retriever_metrics.update(split, retriever_logits, retriever_targets)

    @torch.inference_mode()
    def reset_metrics(self, split: Optional[Split] = None) -> None:
        """
        Reset the metrics corresponding to `split` if provided, else
        reset all the metrics.
        """
        self.reader_metrics.reset(split)
        self.retriever_metrics.reset(split)

    @torch.inference_mode()
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
        self.reader_metrics = self._get_base_metrics(prefix=f"{prefix}answer/")

        self.retriever_metrics = self._get_base_metrics(
            prefix=f"{prefix}document/",
            topk=[1, 3, 5, 10, 20, 50, 100],
            retrieval=True,
            allowed_splits=[Split.VALIDATION, Split.TEST],
        )


@singledispatch
def format_transformer_output(
    output: Any, *, batch: Batch, config: ModelOutputFormatConfig
) -> Batch:
    """Format the output of the transformer to a batch."""
    raise TypeError(f"Cannot handle output of type {type(output)}")


@format_transformer_output.register(CausalLMOutputWithCrossAttentions)
@format_transformer_output.register(CausalLMOutputWithPast)
def _(
    output: CausalLMOutputWithCrossAttentions | CausalLMOutputWithPast,
    *,
    batch: Batch,
    config: ModelOutputFormatConfig,
) -> Batch:
    try:
        lm_logits = output.logits
    except AttributeError as exc:
        raise ValueError(
            f"The output of the transformer must have a `logits` attribute. "
            f"Found {dict(output).keys()}"
        ) from exc

    # get the tensors
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    token_type_ids = batch.get("token_type_ids", None)

    # Shift so that tokens < n predict n
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_input_ids = input_ids[..., 1:].contiguous()
    shift_attention_mask = attention_mask[..., 1:].contiguous()
    if token_type_ids is not None:
        shift_token_type_ids = token_type_ids[..., 1:].contiguous()
    else:
        shift_token_type_ids = None

    # extract the logits of the answer tokens
    if config.answer_choices_tokens is not None:
        logits_mc_answers = extract_mc_answer_logits(
            shift_logits,
            shift_input_ids,
            shift_token_type_ids,
            config.answer_choices_tokens,
            config=config,
        )
    else:
        logits_mc_answers = None

    # Flatten the tokens and compute log `p(x[1:T])`
    loss_fct = CrossEntropyLoss(reduction="none")
    logp_tokens = -loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_input_ids.view(-1))
    logp_tokens = logp_tokens.view_as(shift_attention_mask)
    logp_tokens = logp_tokens * shift_attention_mask

    # compute the loss terms
    output = {"logp": logp_tokens.sum(dim=-1)}
    if token_type_ids is not None:
        output.update(
            {
                "logp_q": (logp_tokens * (shift_token_type_ids == 0).float()).sum(dim=1),
                "logp_a": (logp_tokens * (shift_token_type_ids == 1).float()).sum(dim=1),
            }
        )
    if logits_mc_answers is not None:
        output["mc_logits"] = logits_mc_answers

    return output


def extract_mc_answer_logits(logits, input_ids, token_type_ids, answer_choices_tokens, config):
    """infer the logits of the multiple choice answer tokens
    this corresponds to the logits of the tokens A,B,C,D at the beginning of
    the answer (`token_type_ids == 1`)."""

    # find the index of the first answer token
    # ids = torch.arange(0, token_type_ids.shape[1], device=token_type_ids.device)
    # ids = ids[None, :].expand_as(token_type_ids)
    # ids.masked_fill_(mask=token_type_ids == 0, value=1e6)
    # answer_loc = ids.argmin(dim=-1)
    # TODO: fix the above code. The line below (temporary) assumes that all the
    #  answer tokens are at the end of the encoded prompt.
    # answer_loc = token_type_ids.shape[-1] - (token_type_ids == 1).long().sum(dim=-1)

    # correct code?
    ids = torch.arange(input_ids.size(1), device=input_ids.device)[None, :].expand_as(input_ids)
    mask = token_type_ids.clone()
    mask[mask == 1] = -1e9
    answer_loc = 1 + (ids + mask).max(dim=1).values

    # debugging
    # input_ids = input_ids.clone()
    # input_ids.masked_fill(mask=token_type_ids == 0, value=0)
    # for i, x in enumerate(input_ids):
    #     a = answer_loc[i]
    #     ans_str = config.tokenizer.decode(x[a])
    #     rich.print(
    #         f"Target tokens: (L={token_type_ids[i].sum()}) `{config.tokenizer.decode(x[a:])}` "
    #         f" +  tokens_ans: `{ans_str}`")
    #     if ans_str not in "ABCD":
    #         rich.print(f"[magenta] ERROR -> {ans_str}")

    # fetch the corresponding logits
    # rich.print(f"[green] answer_loc: {answer_loc}")
    answer_loc = answer_loc[:, None, None]
    answer_loc = answer_loc.expand(-1, 1, logits.shape[-1])
    logits_mc_answers = logits.gather(1, index=answer_loc).squeeze(1)

    # preds = logits_mc_answers.argsort(dim=1, descending=True)[:, 3]
    # # for ls in preds:
    # #     rich.print(f"PREDS_AT_ANS: {config.tokenizer.decode(ls, skip_special_tokens=False)}")
    # for i, logits_i in enumerate(logits):
    #     ans_id = answer_loc[i, 0, 0]
    #     preds_i = logits[i].argmax(dim=-1)
    #     preds_i = preds_i[ans_id - 5: ans_id + 5]
    #     rich.print(f">> ({i}, {preds_i})"
    #                f"{config.tokenizer.decode(preds_i, skip_special_tokens=False)}")

    # fetch the logits corresponding to the answer tokens (A, B, C, D)
    mc_tokens_index = answer_choices_tokens[None, :]
    mc_tokens_index = mc_tokens_index.expand(logits_mc_answers.shape[0], -1)
    logits_mc_answers = logits_mc_answers.gather(1, index=mc_tokens_index)
    # rich.print(f">> mc_answer_probs: {logits_mc_answers.softmax(dim=1)}")
    return logits_mc_answers


@format_transformer_output.register(BaseModelOutputWithPoolingAndCrossAttentions)
def _(
    output: BaseModelOutputWithPoolingAndCrossAttentions,
    *,
    batch: Batch,
    config: ModelOutputFormatConfig,
) -> Batch:
    return {
        "pooler_output": output.pooler_output,
    }
