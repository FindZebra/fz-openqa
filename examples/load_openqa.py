import os
import sys

from warp_pipes import pprint_batch
from warp_pipes.support.caching import CacheConfig

from fz_openqa.datamodules.builders.index import IndexBuilder

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
from fz_openqa.datamodules.builders import QaBuilder
from fz_openqa.datamodules.builders import CorpusBuilder
from fz_openqa.datamodules.builders import OpenQaBuilder
from fz_openqa.datamodules.datamodule import DataModule
from fz_openqa.datamodules.pipes import PrioritySampler
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
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if config.get("disable_caching"):
        datasets.disable_caching()
    # datasets.logging.set_verbosity(datasets.logging.CRITICAL)
    transformers.logging.set_verbosity(transformers.logging.CRITICAL)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    seed_everything(1, workers=True)
    # arguments
    num_proc = config.get("num_proc", 4)
    setup_with_model = config.get("setup_with_model", False)
    zero_shot = config.get("zero_shot", True)
    hidden_size = 128
    bert_id = config.get("bert_id", "google/bert_uncased_L-2_H-128_A-2")
    try:
        cache_dir = config.sys.cache_dir
    except Exception:
        cache_dir = default_cache_dir

    base_engine_config = {
        "query_field": "question",
        "index_field": "document",
        "index_key": "row_idx",
        "group_key": "idx",
        "score_key": "proposal_score",
        "verbose": config.get("verbose", False),
    }
    faiss_engines = [
        {
            "name": "dense",
            "config": {
                "k": 100,
                "index_factory": "IVF1,Flat",
                "shard": False,
                **base_engine_config,
            },
        },
    ]

    # load model
    model, zero_shot = load_model(
        config,
        False,
        hidden_size=hidden_size,
        bert_id=bert_id,
        zero_shot=zero_shot,
    )

    # Init Lightning trainer
    logger.info(f"Instantiating trainer <{config.trainer.get('_target_', None)}>")
    trainer: Optional[Trainer] = instantiate(config.trainer)
    if isinstance(trainer, (dict, DictConfig)):
        logger.info("No Trainer was provided. PyTorch Lightning acceleration cannot be used.")
        trainer = None

    # tokenizer and text formatter
    tokenizer = init_pretrained_tokenizer(pretrained_model_name_or_path=bert_id)

    # define the medqa builder
    concat_qa = config.get("concat_qa", False)
    dataset_builder = QaBuilder(
        dset_name=config.get("dset_name", "medqa-us"),
        tokenizer=tokenizer,
        n_query_tokens=1,
        n_answer_tokens=0,
        subset_size=config.get("subset_size", 100),
        add_qad_tokens=not zero_shot,
        cache_dir=cache_dir,
        num_proc=num_proc,
        concat_qa=concat_qa,
    )

    # define the corpus builder
    corpus_builder = CorpusBuilder(
        dset_name=config.get("corpus_name", "findzebra"),
        tokenizer=tokenizer,
        add_qad_tokens=not zero_shot,
        subset_size=config.get("corpus_subset_size", 100),
        append_document_title=True,
        cache_dir=cache_dir,
        num_proc=num_proc,
    )

    # define the index builder
    index_builder = IndexBuilder(
        engines=[
            {
                "name": "elasticsearch",
                "config": {
                    "k": 100,
                    "main_key": "text",
                    "auxiliary_field": "answer",
                    "es_temperature": 5.0,
                    "auxiliary_weight": config.get("aux_weight", 0.5) if concat_qa else 0,
                    **base_engine_config,
                },
            },
            *faiss_engines,
            {
                "name": "topk",
                "config": {"k": 30, "merge_previous_results": False, **base_engine_config},
            },
        ],
        model=model,
        cache_dir=cache_dir,
        index_cache_config=CacheConfig(
            cache_dir=cache_dir,
            model_output_key="_hd_",
            collate_fn=corpus_builder.get_collate_pipe(),
            loader_kwargs={"num_workers": 0, "batch_size": 10},
        ),
        query_cache_config=CacheConfig(
            cache_dir=cache_dir,
            model_output_key="_hq_",
            collate_fn=dataset_builder.get_collate_pipe(nesting_level=0),
            loader_kwargs={"num_workers": 0, "batch_size": 10},
        ),
    )

    # define the OpenQA builder
    builder = OpenQaBuilder(
        dataset_builder=dataset_builder,
        corpus_builder=corpus_builder,
        index_builder=index_builder,
        sampler=PrioritySampler(config={"total": 10}, update=True),
        num_proc=config.get("num_proc", num_proc),
        batch_size=config.get("map_batch_size", 100),
    )

    # define the data module
    dm = DataModule(builder=builder, num_workers=0)

    # preprocess the data
    dm.setup(
        model=model if setup_with_model else None,
        trainer=trainer,
    )
    dm.display_samples(n_samples=1)

    # access dataset
    rich.print(dm.dataset)

    # sample a batch
    _ = next(iter(dm.train_dataloader()))


def load_model(config, use_colbert=False, hidden_size=64, bert_id=None, zero_shot=True):
    if zero_shot:
        model = ZeroShot(
            head="contextual" if use_colbert else "flat", hidden_size=hidden_size, bert_id=bert_id
        )
    else:
        loader = CheckpointLoader(config.get("checkpoint", DEFAULT_CKPT), override=config)
        if config.get("verbose", False):
            loader.print_config()
        model = loader.load_model(last=config.get("last", False))
    model.eval()
    model.freeze()
    return model, zero_shot


if __name__ == "__main__":
    run()
