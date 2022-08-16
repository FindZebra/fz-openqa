import csv
import json
import os
import sys
import warnings
from copy import copy

import torch
from datasets import Split
from elasticsearch.exceptions import ElasticsearchWarning
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode

from fz_openqa.training.training import load_checkpoint
from fz_openqa.utils.elasticsearch import ElasticSearchInstance
from fz_openqa.utils.fingerprint import get_fingerprint

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from loguru import logger
from pathlib import Path
from typing import Optional

import datasets
import hydra
import rich
import transformers
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer

import fz_openqa
from fz_openqa.modeling.zero_shot import ZeroShot
from fz_openqa import configs
from fz_openqa.datamodules.builders import QaBuilder, ConcatQaBuilder
from fz_openqa.datamodules.builders import CorpusBuilder
from fz_openqa.datamodules.builders import OpenQaBuilder
from fz_openqa.datamodules.datamodule import DataModule
from fz_openqa.datamodules.index.builder import IndexBuilder
from fz_openqa.datamodules.pipes import ExactMatch, PrioritySampler
from fz_openqa.inference.checkpoint import CheckpointLoader
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer
from fz_openqa.utils.config import print_config

import fz_openqa.training.experiment  # type: ignore

warnings.filterwarnings(
    action="ignore",
    category=ElasticsearchWarning,
)


@hydra.main(
    config_path=str(Path(configs.__file__).parent),
    config_name="eval_config.yaml",
    version_base="1.2",
)
def run(config):
    """Load the OpenQA dataset mapped using a pre-trained model.

    On the cluster, run:
    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 poetry run python \
    examples/load_openqa.py \
    sys=titan trainer.strategy=dp trainer.gpus=8 +batch_size=1000 +num_workers=10
    ```
    """

    print_config(config)
    # set the context
    datasets.set_caching_enabled(config.get("enable_caching", True))
    # datasets.logging.set_verbosity(datasets.logging.CRITICAL)
    transformers.logging.set_verbosity(transformers.logging.CRITICAL)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    hydra_config = HydraConfig().get()
    seed_everything(1, workers=True)

    # output paths
    work_dir = config.sys.work_dir
    if hydra_config.mode == RunMode.MULTIRUN:
        summary_file = Path(work_dir) / hydra_config.sweep.dir / "results.json"
    else:
        summary_file = Path(work_dir) / hydra_config.run.dir / "results.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    # load the model
    overrides = DictConfig(OmegaConf.to_container(config.overrides, resolve=True))
    logger.info("overriding config with:")
    print_config(overrides)
    checkpoint_manager = load_checkpoint(
        config.checkpoint,
        override_config=overrides,
        ref_config=config,
        silent=True,
    )
    model = checkpoint_manager.load_model(config.checkpoint_type)
    model.eval()
    model.freeze()
    checkpoint_config = checkpoint_manager.config
    logger.info("Checkpoint config:")
    print_config(checkpoint_config)
    logger.info(f"Model fingerprint: {get_fingerprint(model)}")
    exp_id = f"{checkpoint_config.datamodule.dset_name}--{checkpoint_config.datamodule.corpus_name}"

    # load the trainer and dataset
    trainer: Trainer = instantiate(checkpoint_config.trainer)
    # instantiate the datamodule
    logger.info(f"Instantiating datamodule <{checkpoint_config.datamodule._target_}>")
    datamodule: DataModule = instantiate(checkpoint_config.datamodule)
    with ElasticSearchInstance(stdout=open("es.stdout.log", "w")):
        datamodule.setup(trainer=trainer, model=model, clean_caches=False)
    if config.verbose:
        rich.print(datamodule.dataset)
        datamodule.display_samples(n_samples=3, split=config.split[0])

    # evaluate the model
    logits_key = "_reader_logits_"
    all_results = {}
    for split in config.split:
        all_preds = []
        dset = datamodule.dataset[split]
        logger.info(
            f"Evaluating split <{split}> with n={config.n_samples} "
            f"samples ({dset.num_rows} rows)"
        )

        uids = dset["question.uid"]

        for n in range(config.n_samples):
            model.tracked_metrics = {logits_key: None}
            loader = {
                Split.VALIDATION: datamodule.val_dataloader,
                Split.TEST: datamodule.test_dataloader,
            }[split](shuffle=False)
            _ = trainer.test(model=model, dataloaders=loader, verbose=False)
            all_preds.append(model.tracked_metrics[logits_key])

            # print intermediate results
            probs = torch.stack(all_preds, dim=0)
            probs = probs.float().softmax(dim=-1).mean(0)
            preds = probs.argmax(dim=-1)
            if "answer.target" in dset.column_names:
                targets = dset["answer.target"]
            else:
                targets = torch.zeros_like(preds) - 1
            acc = (preds == targets).float().mean().item()
            logger.info(f"{n}/{config.n_samples} - acc={acc:.3%}")

        # gather all predictions
        probs = torch.stack(all_preds, dim=0)
        probs = probs.float().softmax(dim=-1).mean(0)
        preds = probs.argmax(dim=-1)
        if "answer.target" in dset.column_names:
            targets = dset["answer.target"]
        else:
            targets = torch.zeros_like(preds) - 1

        # compute metrics
        acc = (preds == targets).float().mean().item()
        all_results[split] = {"accuracy": acc}
        logger.info(f"# Accuracy (n={config.n_samples}): {acc:.3%}")

        # write the predictions to file
        preds_output_file = Path(f"{split}-preds.txt")
        preds_output_file.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Writing predictions to {preds_output_file.absolute()}")
        with open(preds_output_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "Prediction", "Target", "probs"])
            for i in range(len(preds)):
                writer.writerow([uids[i], preds[i].item(), targets[i].item(), probs[i].tolist()])

    # dump and print all results
    if summary_file.exists():
        summary_data = json.loads(summary_file.read_text())
    else:
        summary_data = {}

    summary_data[exp_id] = all_results
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2)
    rich.print(summary_data)
    logger.info(f"Results stored at {summary_file}")


if __name__ == "__main__":
    run()
