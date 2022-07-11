from __future__ import annotations

import warnings
from copy import copy
from typing import Dict
from typing import Optional

import torch
from omegaconf import DictConfig
from torch import Tensor
from transformers import BertPreTrainedModel
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizerFast

from fz_openqa.utils import maybe_instantiate
from fz_openqa.utils.datastruct import Batch


def instantiate_backbone_model_with_config(
    backbone: DictConfig | PreTrainedModel,
) -> PreTrainedModel:
    """Instantiate a pretrained model and handle loading config objects."""
    if isinstance(backbone, (dict, DictConfig)) and "config" in backbone.keys():
        config = backbone.pop("config", {})
        if len(config) > 0:
            for key in ["_target_", "pretrained_model_name_or_path"]:
                config.pop(key, None)
            config = maybe_instantiate(config)
        msg = (
            "Backbone parameter overrides must be specified in `Config` level, "
            "not in the same level a the main config"
        )
        assert set(backbone.keys()) == {"_target_", "pretrained_model_name_or_path"}, msg
    else:
        config = {}
    backbone: PreTrainedModel = maybe_instantiate(backbone, **config)
    return backbone


def extend_backbone_embeddings(backbone: PreTrainedModel, tokenizer: PreTrainedTokenizerFast):
    """Extend the embeddings to match the vocabulary size of the tokenizer"""
    if backbone.get_input_embeddings().weight.shape[0] != len(tokenizer):
        backbone.resize_token_embeddings(len(tokenizer))

    return backbone


def select_field_attributes(field: str, batch: Batch) -> Batch:
    """Select attributes with prefix `prefix` from the `batch`"""
    prefix = f"{field}."
    return {k.replace(prefix, ""): v for k, v in batch.items() if str(k).startswith(prefix)}


def process_tokens_with_backbone(
    backbone: BertPreTrainedModel,
    input_ids: Tensor,
    attention_mask: Tensor,
    **kwargs,
) -> Tensor:
    backbone_output = backbone(input_ids=input_ids, attention_mask=attention_mask)
    return backbone_output.last_hidden_state


def process_with_backbone(
    batch: Batch,
    *,
    backbone: PreTrainedModel,
    field: Optional[str] = None,
    max_batch_size: Optional[int] = None,
    max_length: int = 512,
    **kwargs,
) -> Tensor | Dict[str, Tensor]:
    batch = copy(batch)

    # select the keys with prefix
    if field is not None:
        batch = select_field_attributes(field, batch)

    if batch["input_ids"].shape[1] > max_length:
        warnings.warn(
            f"Truncating input_ids with "
            f"length={batch['input_ids'].shape[1]} > max_length={max_length}"
        )

    # get input data
    inputs_ids = batch["input_ids"][:, :max_length]
    attention_mask = batch["attention_mask"][:, :max_length]

    # process data by chunk
    output = None
    bs, seq_len = inputs_ids.shape
    if max_batch_size is None:
        chunk_size = bs
    else:
        chunk_size = int(max_batch_size * (max_length / seq_len) ** 2)

    for i in range(0, bs, chunk_size):
        # process the chunk using BERT
        chunk = process_tokens_with_backbone(
            backbone,
            inputs_ids[i : i + chunk_size],
            attention_mask[i : i + chunk_size],
            **kwargs,
        )
        if output is None:
            output = chunk
        else:
            output = torch.cat([output, chunk], dim=0)

    return output
