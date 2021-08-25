import json
import os
from typing import Any
from typing import Dict
from typing import List

import hydra
import pytorch_lightning as pl
import rich
import torch
from datasets import Split
from hydra._internal.instantiate._instantiate2 import _resolve_target
from hydra.utils import instantiate
from omegaconf import DictConfig

from fz_openqa import configs
from fz_openqa.datamodules.fz_x_medqa_dm import FZxMedQADataModule
from fz_openqa.inference.pretty import pprint_results
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer
from fz_openqa.utils.config import print_config
from fz_openqa.utils.config import resolve_config_paths
from fz_openqa.utils.datastruct import pprint_batch
from fz_openqa.utils.train_utils import setup_safe_env


@hydra.main(
    config_path="../configs/",
    config_name="infer_config.yaml",
)
def load_and_infer(config: DictConfig) -> Dict[str, float]:
    """
    Load a train model and process some input data.
    The input data is a source .json file if provided else this is
    the test set of the datamodule described in the config file.

    NB: implemented for a reader model only
    todo: retriever evaluation
    todo: full model evaluation
    todo: loader both a reader and a retriever
    """
    setup_safe_env()

    # replace config paths with loaded configs
    resolve_config_paths(config, path=os.path.dirname(configs.__file__))

    # Pretty print config using Rich library
    if config.get("print_config"):
        print_config(config, resolve=True)

    # load a pretrained model from a checkpoint
    rich.print(
        f">> Instantiating model <{config.model._target_}> "
        f"\n\tfrom checkpoint=`{config.checkpoint_path}`"
        f"\n\ton device=`{config.device}`.."
    )
    cls: pl.LightningModule.__class__ = _resolve_target(config.model._target_)
    model = load_model_from_checkpoint(
        cls,
        path=config.checkpoint_path,
        device=config.device,
        cache_dir=config.cache_dir,
    )
    bert_id: str = model.bert.config._name_or_path

    print(f">> Instantiating tokenizer `{bert_id}`..")
    tokenizer = init_pretrained_tokenizer(
        pretrained_model_name_or_path=bert_id, cache_dir=config.cache_dir
    )

    if config.input_text is None:
        print(f">> Loading datamodule `{config.datamodule._target_}`..")
        dm = instantiate(
            config.datamodule,
            tokenizer=tokenizer,
            cache_dir=config.cache_dir,
            verbose=True,
        )
        dm.prepare_data()
        dm.setup()
        a, b = map(eval, config.slice.split(":"))
        data = dm.text_data[Split.TEST][a:b]
    else:
        print(f">> Loading input data from `{config.input_text}`..")
        data = load_inputs(config.input_text)

    print(">> Encoding input data..")
    batch = encode_data(data, tokenizer)
    pprint_batch(batch)

    print(">> Inferring predictions..")
    output = model(batch)
    pprint_results(data, output)


def encode_data(data, tokenizer):
    """Encode the data retrieve form a .json file"""
    # tokenize
    batch = FZxMedQADataModule.tokenize_examples(
        data, tokenizer=tokenizer, max_length=None
    )
    batch.update(
        **{
            k: v
            for k, v in data.items()
            if k in ["answer.target", "rank", "idx", "is_positive"]
        }
    )
    [batch.pop(k) for k in list(batch.keys()) if ".text" in k]

    # convert as list of batch
    n = len(list(batch.values())[0])
    batch = [{k: batch[k][idx] for k in batch.keys()} for idx in range(n)]

    # tensorize and collate
    batch = [{k: torch.tensor(v) for k, v in d.items()} for d in batch]
    batch = FZxMedQADataModule.collate_fn_(tokenizer, batch)
    return batch


def load_inputs(path: str) -> Dict[str, List[Any]]:
    """Load data from a .json file"""
    with open(path, "r") as fp:
        data = json.load(fp)

    # convert as dict of list
    keys = list(data[0].keys())
    return {k: [d[k] for d in data] for k in keys}


def load_model_from_checkpoint(
    cls: pl.LightningModule.__class__,
    *,
    path: str,
    device: torch.device,
    **kwargs,
):
    """load the model form a checkpoint"""
    model = cls.load_from_checkpoint(path, map_location=device, **kwargs)
    model.eval()
    model.freeze()
    return model
