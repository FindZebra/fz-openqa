import math
import os
import sys

import numpy as np
import requests
from tqdm import tqdm

from fz_openqa.training.training import load_checkpoint
from fz_openqa.utils.elasticsearch import ElasticSearchInstance
from fz_openqa.utils.fingerprint import get_fingerprint

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import logging
from pathlib import Path

import datasets
import hydra
import rich
import transformers
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer

from fz_openqa import configs
from fz_openqa.datamodules.datamodule import DataModule
from fz_openqa.utils.config import print_config

logger = logging.getLogger(__name__)

import fz_openqa.training.experiment  # type: ignore


def get_rank(x, y):
    if x not in y:
        return math.inf
    return 1 + min([i for i, yy in enumerate(y) if yy == x])


def reduce_docs(docs, doc_ids):
    new_docs = []
    selected_doc_ids = set()
    for d, idx in zip(docs, doc_ids):
        if idx not in selected_doc_ids:
            new_docs.append(d)
            selected_doc_ids.add(idx)
    return new_docs


def get_fz_rank(query, que_cuis):
    base_url = "https://www.findzebra.com/api/v1/query"
    params = {
        "q": query,
        "response_format": "json",
        "rows": 100,
        "api_key": "2bde80df-c408-4a15-acec-dfd0201812ce",
    }
    response = requests.get(base_url, params=params)
    content = response.json()
    docs = content["response"]["docs"]
    doc_cuis = [d.get("cui", "--") for d in docs]
    doc_cuis = map(str.lower, doc_cuis)
    doc_cuis = list(doc_cuis)
    # doc_cuis = reduce_docs(doc_cuis)
    return min(get_rank(c, doc_cuis) for c in que_cuis)


@hydra.main(
    config_path=str(Path(configs.__file__).parent),
    config_name="eval_retrieval_config.yaml",
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
    datasets.set_caching_enabled(True)
    datasets.logging.set_verbosity(datasets.logging.CRITICAL)
    transformers.logging.set_verbosity(transformers.logging.CRITICAL)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_DATASETS_CACHE"] = config.sys.cache_dir
    seed_everything(1, workers=True)

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
    checkpoint_config = checkpoint_manager.config
    logger.info("Checkpoint config:")
    print_config(checkpoint_config)
    logger.info(f"Model fingerprint: {get_fingerprint(model)}")

    # load the trainer and dataset
    trainer: Trainer = instantiate(checkpoint_config.trainer)
    # instantiate the datamodule
    logger.info(f"Instantiating datamodule <{checkpoint_config.datamodule._target_}>")
    datamodule: DataModule = instantiate(checkpoint_config.datamodule)
    rich.print(datamodule)
    with ElasticSearchInstance(stdout=open("es.stdout.log", "w")):
        datamodule.setup(trainer=trainer, model=model, clean_caches=False)

    # iterate batch and compute metrics
    ranks = []
    fz_ranks = []
    output_file = Path("output.txt")
    rich.print(f"Writing output to {output_file.absolute()}")
    j = 0
    with open(output_file, "w") as f:
        for batch in tqdm(datamodule.test_dataloader(), desc="Evaluating"):
            for i in range(len(batch["document.cui"])):
                j += 1
                eg = {k: v[i] for k, v in batch.items()}
                doc_cuis = map(str.lower, eg["document.cui"])
                doc_ids = eg["document.idx"]
                doc_cuis = reduce_docs(doc_cuis, doc_ids)
                que_cuis = list(map(str.lower, eg["question.cui"]))

                # compute the rank
                rank = min(get_rank(c, list(doc_cuis)) for c in que_cuis)
                ranks.append(rank)
                metrics = get_metrics(ranks)

                # compute the fz rank (baseline)
                fz_rank = get_fz_rank(eg["question.text"], que_cuis)
                fz_ranks.append(fz_rank)
                fz_metrics = get_metrics(fz_ranks)

                # print the intermediate results
                desc = ", ".join(f"{k}={v:.3f}" for k, v in metrics.items())
                desc += f", FZ:MRR={fz_metrics.get('MRR', -1):.3f}"
                rich.print(desc)

                # write to file
                f.write(f"Question #{j} - deep:rank={rank}, api:rank={fz_rank}\n\n")
                f.write(f"{eg['question.text']}\n\n")
                f.write(100 * "-" + "\n")
                for n, doc in enumerate(eg["document.text"][:5]):
                    title = eg["document.title"][n]
                    f.write(f"Passage #{n + 1}, Title: {title}.\n{doc}\n")
                    f.write(100 * "." + "\n")
                f.write(100 * "=" + "\n")

    if config.verbose:
        datamodule.display_samples(n_samples=10, split=config.split[0])

    metrics = get_metrics(ranks)
    fz_metrics = get_metrics(fz_ranks)
    rich.print(
        {
            "VODQA": metrics,
            "FZ": fz_metrics,
        }
    )
    rich.print(f"Output file: {output_file.absolute()}")


def get_metrics(ranks):
    output = {
        "MRR": np.mean(1.0 / np.array(ranks)),
    }
    for t in [3, 5, 10, 20, 50, 100]:
        m = np.mean([1 if r <= t else 0 for r in ranks])
        output[f"Hit@{t}"] = m

    return output


if __name__ == "__main__":
    run()
