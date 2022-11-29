import os
from typing import Type

import torch
from hydra._internal.instantiate._instantiate2 import _resolve_target
from loguru import logger
from optimum.bettertransformer import BetterTransformer
from transformers import AutoModel
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer


def extend_vocabulary_fn(
    model: PreTrainedModel, *, tokenizer: PreTrainedTokenizer
) -> PreTrainedModel:
    if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        logger.info(
            f"Extending vocabulary of model {type(model)} from "
            f"size {model.get_input_embeddings().weight.shape[0]} "
            f"to {len(tokenizer)}."
        )
        model.resize_token_embeddings(len(tokenizer))
    return model


def init_pretrained_model(
    model_id: str,
    *,
    Cls: Type[PreTrainedModel] = AutoModel,
    tokenizer: PreTrainedTokenizer,
    extend_vocabulary: bool = True,
    use_optimum: bool = False,
    **kwargs,
) -> PreTrainedModel:
    """Load a pretrained model from the HuggingFace model hub."""
    if "DISABLE_OPTIMUM" in os.environ:
        logger.info("Optimum is disabled.")
        use_optimum = False

    if isinstance(Cls, str):
        Cls = _resolve_target(Cls, Cls)
    model = Cls.from_pretrained(model_id, **kwargs)
    if extend_vocabulary:
        model = extend_vocabulary_fn(model, tokenizer=tokenizer)
    if use_optimum and "vod-retriever-medical-v1.0" not in model_id:
        try:
            model = BetterTransformer.transform(model)
        except Exception as exc:
            logger.warning(f"Failed to transform model {model_id} to optimum model: {exc}")
    model = handle_special_cases(model_id, model)
    return model


def handle_special_cases(model_id: str, model: PreTrainedModel) -> PreTrainedModel:
    if "findzebra/vod-retriever-medical-v1.0" == model_id:
        # temporary hack: the VOD model was trained without activation layer
        # between the body and the pooler layer. Replacing the pooler layer
        # before uploading the model to the hub, didn't work. So we add the
        # activation layer here as a temporary fix.
        model.pooler.activation = torch.nn.Identity()

    return model
