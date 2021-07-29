import json
import os
from typing import Any
from typing import Dict
from typing import List

import hydra
import pytorch_lightning as pl
import rich
import torch
from hydra._internal.instantiate._instantiate2 import _resolve_target
from omegaconf import DictConfig
from rich.console import Console
from rich.table import Table

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

    print(f">> here: {os.listdir()}")

    # load model
    rich.print(
        f">> Instantiating model <{config.model._target_}> "
        f"\n\tfrom checkpoint=`{config.checkpoint_path}`"
        f"\n\ton device=`{config.device}`.."
    )
    cls: pl.LightningModule.__class__ = _resolve_target(config.model._target_)
    model = load_model(
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

    print(f">> Loading input data from `{config.input_text}`..")
    data = load_inputs(config.input_text)

    print(">> Encoding input data..")
    batch = encode_data(data, tokenizer)
    pprint_batch(batch)

    print(">> Processing input data..")
    output = model(batch)

    pprint_results(data, output)


def pprint_results(data, output):
    table = Table(title="Results", show_lines=True)
    table.add_column("Question", justify="left", style="cyan", no_wrap=False)
    table.add_column(
        "Document", justify="left", style="magenta", no_wrap=False
    )
    for idx in range(len(data[0]["answer_choices"])):
        table.add_column(f"Answer {idx}", justify="center", style="white")
    for idx, row in enumerate(data):
        probs = output[idx].softmax(-1)
        pred = probs.argmax(-1)
        table.add_row(
            row["question"],
            row["document"],
            *(
                format_ans_prob(k, a, row["answer_idx"], pred, p)
                for k, (a, p) in enumerate(zip(row["answer_choices"], probs))
            ),
        )
    console = Console()
    console.print(table)


def format_ans_prob(k, txt, adx, pred, p):
    u = f"{txt}"
    if adx == k:
        u = f"[bold underline]{u}[/underline bold]"

    u += f"\n{100 * p:.2f}%"

    if k == adx and adx == pred:
        u = f"[green]{u}[/green]"
    elif k == pred:
        u = f"[red]{u}[/red]"

    return u


def encode_data(data, tokenizer):
    # convert as dict of list
    keys = list(data[0].keys())
    data = {k: [d[k] for d in data] for k in keys}

    # tokenize
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

    # convert as list of batch
    n = len(list(batch.values())[0])
    batch = [{k: batch[k][idx] for k in batch.keys()} for idx in range(n)]

    # tensorize and collate
    batch = [{k: torch.tensor(v) for k, v in d.items()} for d in batch]
    batch = FZxMedQADataModule.collate_fn_(tokenizer, batch)
    return batch


def load_inputs(path: str) -> List[Dict[str, Any]]:
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
