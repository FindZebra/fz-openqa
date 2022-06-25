import os
import sys

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
from fz_openqa.datamodules.builders import QaBuilder, ConcatQaBuilder
from fz_openqa.datamodules.builders import CorpusBuilder
from fz_openqa.datamodules.builders import OpenQaBuilder
from fz_openqa.datamodules.datamodule import DataModule
from fz_openqa.datamodules.index.builder import IndexBuilder
from fz_openqa.datamodules.pipes import ExactMatch, PrioritySampler
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
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 poetry run python \
    examples/load_openqa.py \
    sys=titan trainer.strategy=dp trainer.gpus=8 +batch_size=1000 +num_workers=10
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
    num_proc = config.get("num_proc", 4)
    setup_with_model = config.get("setup_with_model", False)
    hidden_size = 32 if use_colbert else 128
    bert_id = config.get("bert_id", "google/bert_uncased_L-2_H-128_A-2")
    try:
        cache_dir = config.sys.cache_dir
    except Exception:
        cache_dir = default_cache_dir

    # faiss engines
    if use_colbert:
        faiss_engines = [
            {
                "name": "maxsim",
                "k": 100,
                "merge_previous_results": False,
                "max_batch_size": 100,
                "config": {
                    "max_chunksize": 1000,
                },
            },
        ]
    else:
        faiss_engines = [
            {
                "name": "faiss",
                "k": 100,
                "merge_previous_results": False,
                "max_batch_size": 100,
                "config": {
                    "index_factory": "IVF1000,Flat",
                },
            },
        ]

    # load model
    model, zero_shot = load_model(config, use_colbert, hidden_size=hidden_size, bert_id=bert_id)

    # Init Lightning trainer
    logger.info(f"Instantiating trainer <{config.trainer.get('_target_', None)}>")
    trainer: Optional[Trainer] = instantiate(config.trainer)
    if isinstance(trainer, (dict, DictConfig)):
        logger.info("No Trainer was provided. PyTorch Lightning acceleration cannot be used.")
        trainer = None

    # tokenizer and text formatter
    tokenizer = init_pretrained_tokenizer(pretrained_model_name_or_path=bert_id)

    # define the medqa builder
    concat_dset = config.get("concat_qa", True)
    QaBuilderCls = ConcatQaBuilder if concat_dset else QaBuilder
    dataset_builder = QaBuilderCls(
        dset_name=config.get("dset_name", "medqa-us"),
        tokenizer=tokenizer,
        n_query_tokens=1,
        n_answer_tokens=0,
        subset_size=1000 if config.get("use_subset", True) else None,
        add_qad_tokens=not zero_shot or not setup_with_model,
        cache_dir=cache_dir,
        num_proc=num_proc,
    )

    # define the corpus builder
    corpus_builder = CorpusBuilder(
        dset_name=config.get("corpus_name", "medwiki"),
        tokenizer=tokenizer,
        add_qad_tokens=not zero_shot or not setup_with_model,
        subset_size=100 if config.get("corpus_subset", False) else None,
        cache_dir=cache_dir,
        num_proc=num_proc,
    )

    # define the index builder
    index_builder = IndexBuilder(
        engines=[
            {
                "name": "es",
                "k": 100,
                "merge_previous_results": True,
                "max_batch_size": 512,
                "verbose": False,
                "config": {
                    "es_temperature": 5.0,
                    "auxiliary_weight": config.get("aux_weight", 0.5) if concat_dset else 0,
                    "filter_with_doc_ids": False,
                },
            },
            *faiss_engines,
            {
                "name": "topk",
                "k": 100,
                "max_batch_size": 100,
            },
        ],
        corpus_collate_pipe=corpus_builder.get_collate_pipe(),
        loader_kwargs={
            "batch_size": config.get("batch_size", 10),
            "num_workers": config.get("num_workers", 4),
            "pin_memory": config.get("pin_memory", True),
        },
        cache_dir=cache_dir,
        persist_cache=False,
    )

    # define the OpenQA builder
    builder = OpenQaBuilder(
        dataset_builder=dataset_builder,
        corpus_builder=corpus_builder,
        index_builder=index_builder,
        relevance_classifier=ExactMatch(interpretable=True),
        sampler=PrioritySampler(total=10),
        n_retrieved_documents=100,
        num_proc=config.get("num_proc", num_proc),
        batch_size=config.get("map_batch_size", 100),
    )

    # define the data module
    dm = DataModule(builder=builder)

    # preprocess the data
    dm.setup(
        model=model if setup_with_model else None,
        trainer=trainer,
    )
    dm.display_samples(n_samples=10)

    # access dataset
    rich.print(dm.dataset)

    # sample a batch
    # _ = next(iter(dm.train_dataloader()))


def load_model(config, use_colbert, hidden_size=64, bert_id=None):
    zero_shot = config.get("zero_shot", True)
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
