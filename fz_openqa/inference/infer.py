import json
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
from fz_openqa.datamodules.fz_x_medqa_dm import FZxMedQADataModule
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer
from fz_openqa.utils.config import print_config
from fz_openqa.utils.config import resolve_config_paths
from fz_openqa.utils.datastruct import pprint_batch


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

    print(f">> Instantiating tokenizer `{bert_id}`")
    tokenizer = init_pretrained_tokenizer(
        pretrained_model_name_or_path=bert_id, cache_dir=config.cache_dir
    )
    rich.print(tokenizer)

    print(f">> Loading input data from `{config.input_text}`")
    data = load_inputs(config.input_text)
    rich.print(data)

    print(">> Tokenizing input data")
    data = {k: [v] for k, v in data.items()}
    batch = FZxMedQADataModule.tokenize_examples(
        data, tokenizer=tokenizer, max_length=None
    )
    batch.update(
        **{
            k: v
            for k, v in data.items()
            if k in ["answer_idx", "rank", "idx", "is_positive"]
        }
    )
    [batch.pop(k) for k in list(batch.keys()) if ".text" in k]
    batch = {k: torch.tensor(v[0]) for k, v in batch.items()}
    batch = FZxMedQADataModule.collate_fn_(tokenizer, [batch])
    pprint_batch(batch)

    print(">> Processing input data")
    output = model(batch)
    rich.print(output.softmax(-1))


def load_inputs(path: str):
    with open(path, "r") as fp:
        return json.load(fp)


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
