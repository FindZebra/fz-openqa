import os
import sys

import faiss

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import logging
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
from fz_openqa.datamodules.builders import MedQaBuilder
from fz_openqa.datamodules.builders import MedQaCorpusBuilder
from fz_openqa.datamodules.builders import OpenQaBuilder
from fz_openqa.datamodules.datamodule import DataModule
from fz_openqa.datamodules.index.builder import FaissIndexBuilder, ColbertIndexBuilder
from fz_openqa.datamodules.pipes import ExactMatch
from fz_openqa.datamodules.pipes import TextFormatter
from fz_openqa.inference.checkpoint import CheckpointLoader
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer
from fz_openqa.utils.config import print_config

logger = logging.getLogger(__name__)

DEFAULT_CKPT = "https://drive.google.com/file/d/17XDASu1JYGndCFWNW3zCZGKD2woJuDIg/view?usp=sharing"
default_cache_dir = Path(fz_openqa.__file__).parent.parent / "cache"

OmegaConf.register_new_resolver("whoami", lambda: os.environ.get("USER"))
OmegaConf.register_new_resolver("getcwd", os.getcwd)


@hydra.main(
    config_path=str(Path(configs.__file__).parent),
    config_name="script_config.yaml",
)
def run(config):
    """Load the OpenQA dataset mapped using a pre-trained model.

    On the cluster, run:
    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 poetry run python examples/load_mapped_medqa_faiss.py \
    sys=titan trainer.strategy=dp trainer.gpus=8 +batch_size=1000 \
    +num_workers=10 +use_subset=False +colbert=True

    ```
    """
    print_config(config)
    # set the context
    datasets.set_caching_enabled(True)
    # datasets.logging.set_verbosity(datasets.logging.CRITICAL)
    transformers.logging.set_verbosity(transformers.logging.CRITICAL)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    seed_everything(1, workers=True)
    use_colbert = config.get("colbert", False)
    try:
        cache_dir = config.sys.cache_dir
    except Exception:
        cache_dir = default_cache_dir

    # load model
    zero_shot = config.get("zero_shot", True)
    if zero_shot:
        model = ZeroShot(head="contextual" if use_colbert else "flat", limit_size=32)
    else:
        loader = CheckpointLoader(config.get("checkpoint", DEFAULT_CKPT), override=config)
        if config.get("verbose", False):
            loader.print_config()
        model = loader.load_model(last=config.get("last", False))
    model.eval()
    model.freeze()

    # Init Lightning trainer
    logger.info(f"Instantiating trainer <{config.trainer.get('_target_', None)}>")
    trainer: Optional[Trainer] = instantiate(config.trainer)
    if isinstance(trainer, (dict, DictConfig)):
        logger.info("No Trainer was provided. PyTorch Lightning acceleration cannot be used.")
        trainer = None

    # tokenizer and text formatter
    tokenizer = init_pretrained_tokenizer(
        pretrained_model_name_or_path="dmis-lab/biobert-base-cased-v1.2"
    )
    text_formatter = TextFormatter(lowercase=True)

    # define the medqa builder
    dataset_builder = MedQaBuilder(
        tokenizer=tokenizer,
        text_formatter=text_formatter,
        use_subset=config.get("use_subset", True),
        add_encoding_tokens=not zero_shot,
        cache_dir=cache_dir,
        num_proc=4,
    )
    dataset_builder.subset_size = [200, 50, 50]

    # define the corpus builder
    corpus_builder = MedQaCorpusBuilder(
        tokenizer=tokenizer,
        text_formatter=text_formatter,
        add_encoding_tokens=not zero_shot,
        use_subset=config.get("corpus_subset", False),
        cache_dir=cache_dir,
        num_proc=4,
    )

    # define the index builder
    IndexCls = ColbertIndexBuilder if use_colbert else FaissIndexBuilder

    faiss_args = {
        "type": "IVFQ",
        "metric_type": faiss.METRIC_INNER_PRODUCT,
        "n_list": 100,
        "n_subvectors": 16,
        "n_bits": 8,
        "nprobe": 16,
    }

    index_builder = IndexCls(
        model=model,
        trainer=trainer,
        model_output_keys=["_hd_", "_hq_"],
        collate_pipe=corpus_builder.get_collate_pipe(),
        loader_kwargs={
            "batch_size": config.get("batch_size", 10),
            "num_workers": config.get("num_workers", 4),
            "pin_memory": config.get("pin_memory", True),
        },
        cache_dir=cache_dir,
        persist_cache=True,
        progress_bar=True,
        faiss_train_size=1000 if use_colbert else 10000,
        faiss_args=faiss_args,
        in_memory=config.get("in_memory", True),
    )

    # define the OpenQA builder
    builder = OpenQaBuilder(
        dataset_builder=dataset_builder,
        corpus_builder=corpus_builder,
        index_builder=index_builder,
        relevance_classifier=ExactMatch(interpretable=True),
        n_retrieved_documents=config.get("n_retrieved_documents", 1000),
        n_documents=10,
        max_pos_docs=1,
        filter_unmatched=True,
        num_proc=config.get("num_proc", 4),
        batch_size=config.get("map_batch_size", 8000),
        select_mode=config.get("select_mode", "sample"),
    )

    # define the data module
    dm = DataModule(builder=builder)

    # preprocess the data
    dm.prepare_data()
    dm.setup()
    dm.display_samples(n_samples=3)

    # access dataset
    rich.print(dm.dataset)

    # sample a batch
    # _ = next(iter(dm.train_dataloader()))


if __name__ == "__main__":
    run()
