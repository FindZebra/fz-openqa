from __future__ import annotations

from omegaconf import DictConfig
from transformers import BertPreTrainedModel

from fz_openqa.utils import maybe_instantiate


def instantiate_backbone_model_with_config(
    bert: DictConfig | BertPreTrainedModel,
) -> BertPreTrainedModel:
    if isinstance(bert, (dict, DictConfig)) and "config" in bert.keys():
        bert_config = bert.pop("config", {})
        if len(bert_config) > 0:
            for key in ["_target_", "pretrained_model_name_or_path"]:
                bert_config.pop(key, None)
            bert_config = maybe_instantiate(bert_config)
        msg = (
            "BERT parameter overrides must be specified in `BertConfig`, " "not in the model config"
        )
        assert set(bert.keys()) == {"_target_", "pretrained_model_name_or_path"}, msg
    else:
        bert_config = {}
    bert: BertPreTrainedModel = maybe_instantiate(bert, **bert_config)
    return bert
