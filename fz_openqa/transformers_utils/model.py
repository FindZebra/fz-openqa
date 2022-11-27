from typing import Type

import rich
import torch
from hydra._internal.instantiate._instantiate2 import _resolve_target
from loguru import logger
from transformers import AutoModel
from transformers import AutoModelForCausalLM
from transformers import AutoModelForSequenceClassification
from transformers import GPT2LMHeadModel
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer


def extend_vocabulary(model: PreTrainedModel, *, tokenizer: PreTrainedTokenizer) -> PreTrainedModel:
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
    **kwargs,
) -> PreTrainedModel:
    """Load a pretrained model from the HuggingFace model hub."""
    if isinstance(Cls, str):
        Cls = _resolve_target(Cls, Cls)
    model = Cls.from_pretrained(model_id, **kwargs)
    model = extend_vocabulary(model, tokenizer=tokenizer)
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
