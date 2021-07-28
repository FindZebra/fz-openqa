import os
from typing import Dict

import hydra
import pytorch_lightning as pl
import rich
import torch
from hydra._internal.instantiate._instantiate2 import _resolve_target
from omegaconf import DictConfig
from transformers import PreTrainedModel

from fz_openqa import configs
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer
from fz_openqa.utils.config import print_config
from fz_openqa.utils.config import resolve_config_paths


@hydra.main(
    config_path="../configs/",
    config_name="infer_config.yaml",
)
def load_and_infer(config: DictConfig) -> Dict[str, float]:
    # replace config paths with loaded configs
    resolve_config_paths(config, path=os.path.dirname(configs.__file__))

    # Pretty print config using Rich library
    if config.get("print_config"):
        print_config(config, resolve=True)

    # load model
    rich.print(
        f">> Instantiating model <{config.model._target_}> "
        f"\n\tfrom checkpoint=`{config.checkpoint_path}`"
        f"\n\ton device=`{config.device}`"
    )
    cls: pl.LightningModule.__class__ = _resolve_target(config.model._target_)
    model = load_model(
        cls,
        path=config.checkpoint_path,
        device=config.device,
        cache_dir=config.cache_dir,
    )
    bert_id: str = model.bert.config._name_or_path
    print(">> model successfully loaded!")
    rich.print(model.bert.config)

    print(f">> Instantiating tokenizer `{bert_id}` ...")
    tokenizer = init_pretrained_tokenizer(
        pretrained_model_name_or_path=bert_id, cache_dir=config.cache_dir
    )
    rich.print(tokenizer)


def load_model(
    cls: pl.LightningModule.__class__,
    *,
    path: str,
    device: torch.device,
    **kwargs,
):
    model = cls.load_from_checkpoint(path, map_location=device, **kwargs)
    model.eval()
    model.freeze()
    return model
