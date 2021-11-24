import logging
import os
from pathlib import Path
from typing import Callable
from typing import Iterable
from typing import Optional

import datasets
import hydra
import pytorch_lightning
import rich
import torch
import transformers
from datasets import Dataset
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning import Callback
from pytorch_lightning import LightningModule
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import move_data_to_device
from rich.progress import track
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler

import fz_openqa
from fz_openqa import configs
from fz_openqa.callbacks.store_results import StorePredictionsCallback
from fz_openqa.datamodules.builders.corpus import MedQaCorpusBuilder
from fz_openqa.datamodules.index import FaissIndex
from fz_openqa.datamodules.index.dense import AddRowIdx
from fz_openqa.datamodules.index.pipes import FetchNestedDocuments
from fz_openqa.datamodules.pipes import Parallel
from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.datamodules.pipes import SearchCorpus
from fz_openqa.datamodules.pipes.predict import Predict
from fz_openqa.inference.checkpoint import CheckpointLoader
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.fingerprint import get_fingerprint
from fz_openqa.utils.functional import infer_batch_size
from fz_openqa.utils.pretty import get_separator
from fz_openqa.utils.pretty import pprint_batch

logger = logging.getLogger(__name__)

DEFAULT_CKPT = "https://drive.google.com/file/d/17XDASu1JYGndCFWNW3zCZGKD2woJuDIg/view?usp=sharing"
# define the default cache location
default_cache_dir = Path(fz_openqa.__file__).parent.parent / "cache"

OmegaConf.register_new_resolver("whoami", lambda: os.environ.get("USER"))
OmegaConf.register_new_resolver("getcwd", os.getcwd)

import pytorch_lightning as pl


@torch.no_grad()
@hydra.main(
    config_path=str(Path(configs.__file__).parent),
    config_name="script_config.yaml",
)
def run(config: DictConfig) -> None:
    """
    Load a corpus and index it using Faiss.
    Then query the corpus using the 3 first corpus documents.

    PytorchLightning's Trainer can be used to accelerate indexing.
    Example, to index the whole corpus (~5min on `rhea`):
    ```bash
    poetry run python examples/index_corpus_using_dense_retriever.py \
    trainer.strategy=dp trainer.gpus=8 +batch_size=2000 +num_workers=16
    +n_samples=null +use_subset=False +num_proc=4

    ```
    """
    # set the context
    datasets.set_caching_enabled(True)
    datasets.logging.set_verbosity(datasets.logging.CRITICAL)
    transformers.logging.set_verbosity(transformers.logging.CRITICAL)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    seed_everything(1, workers=True)
    cache_dir = config.get("sys.cache_dir", default_cache_dir)

    # load model
    loader = CheckpointLoader(config.get("checkpoint", DEFAULT_CKPT), override=config)
    if config.get("verbose", False):
        loader.print_config()
    model = loader.load_model(last=config.get("last", False))
    model.eval()
    model.freeze()
    logger.info(f"Model {type(model)} loaded")
    logger.info(f"Model fingerprint: {get_fingerprint(model.module.bert)}")

    # Init Lightning trainer
    logger.info(f"Instantiating trainer <{config.trainer.get('_target_', None)}>")
    trainer: Optional[Trainer] = instantiate(config.trainer)
    if isinstance(trainer, (dict, DictConfig)):
        logger.info("No Trainer was provided. PyTorch Lightning acceleration cannot be used.")
        trainer = None

    # set up the corpus builder
    logger.info(f"Initialize corpus <{MedQaCorpusBuilder.__name__}>")
    corpus_builder = MedQaCorpusBuilder(
        tokenizer=loader.tokenizer,
        to_sentences=config.get("to_sentences", False),
        use_subset=config.get("use_subset", True),
        cache_dir=cache_dir,
        num_proc=config.get("num_proc", 2),
    )

    # build the corpus and take a subset
    corpus = corpus_builder()
    n_samples = config.get("n_samples", 100)
    if n_samples is not None and n_samples > 0:
        n_samples = min(n_samples, len(corpus))
        corpus = corpus.select(range(n_samples))
    rich.print(corpus)

    # init the pipe
    predict = Predict(model=model)
    predict.cache(
        corpus,
        trainer=trainer,
        collate_fn=corpus_builder.get_collate_pipe(),
        cache_dir=cache_dir,
        loader_kwargs={"batch_size": 2},
        persist=True,
    )

    rich.print(f">> cache_file={predict.cache_file}")

    idx = list(range(n_samples))
    batch = corpus_builder.get_collate_pipe()([corpus[i] for i in idx])

    # idx = list(range(n_samples + 1))
    rich.print("[green]>> processing with cache")
    pprint_batch(predict(batch, idx=idx))

    rich.print("[green]>> processing with cache -- non contiguous")
    pprint_batch(predict(batch, idx=[2, 0, 1]))

    rich.print("[red]>> processing without cache")
    pprint_batch(predict(batch, idx=None))

    rich.print("[cyan]>> Reading the cache in Dataset.map()")
    _ = corpus.map(predict, with_indices=True, batched=True, batch_size=3)


if __name__ == "__main__":
    run()
