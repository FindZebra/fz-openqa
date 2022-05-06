from typing import Dict

import rich
import transformers
from loguru import logger
from transformers import AutoConfig
from transformers import AutoModel
from transformers import BertModel
from transformers import PretrainedConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.models.auto.auto_factory import _get_model_class

from fz_openqa.modeling.bert.custom_bert import CustomBertModel

CUSTOM_TRANSFORMERS = {BertModel: CustomBertModel}


class CustomAutoModel(AutoModel):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        trust_remote_code = kwargs.pop("trust_remote_code", False)
        kwargs["_from_auto"] = True
        if not isinstance(config, PretrainedConfig):
            config, kwargs = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                return_unused_kwargs=True,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )
        if hasattr(config, "auto_map") and cls.__name__ in config.auto_map:
            if not trust_remote_code:
                raise ValueError(
                    f"Loading {pretrained_model_name_or_path} requires you to execute the modeling file in that repo "
                    "on your local machine. Make sure you have read the code there to avoid malicious use, then set "
                    "the option `trust_remote_code=True` to remove this error."
                )
            if kwargs.get("revision", None) is None:
                logger.warning(
                    "Explicitly passing a `revision` is encouraged when loading a model with custom code to ensure "
                    "no malicious code has been contributed in a newer revision."
                )
            class_ref = config.auto_map[cls.__name__]
            module_file, class_name = class_ref.split(".")
            model_class = get_class_from_dynamic_module(
                pretrained_model_name_or_path, module_file + ".py", class_name, **kwargs
            )

            # replace the original model class with a custom one
            model_class = cls._replace_model_class_with_custom(model_class)

            return model_class.from_pretrained(
                pretrained_model_name_or_path, *model_args, config=config, **kwargs
            )
        elif type(config) in cls._model_mapping.keys():
            model_class = _get_model_class(config, cls._model_mapping)

            # replace the original model class with a custom one
            model_class = cls._replace_model_class_with_custom(model_class)

            return model_class.from_pretrained(
                pretrained_model_name_or_path, *model_args, config=config, **kwargs
            )
        raise ValueError(
            f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}.\n"
            f"Model type should be one of {', '.join(c.__name__ for c in cls._model_mapping.keys())}."
        )

    @classmethod
    def _replace_model_class_with_custom(cls, model_class):
        if model_class in CUSTOM_TRANSFORMERS:
            logger.warning(
                f"Replacing original class <{model_class.__name__}> with"
                f"custom class {CUSTOM_TRANSFORMERS[model_class]}"
            )
            model_class = CUSTOM_TRANSFORMERS[model_class]
        return model_class
