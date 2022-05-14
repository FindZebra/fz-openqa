import os
import sys

from fz_openqa.datamodules.builders import QaBuilder
from fz_openqa.datamodules.index.base import IndexMode
from fz_openqa.datamodules.index.colbert import ColbertIndex
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import logging
from pathlib import Path
from typing import Optional

import datasets
import faiss
import hydra
import rich
import torch
import transformers
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer

import fz_openqa
from fz_openqa.modeling.zero_shot import ZeroShot
from fz_openqa import configs
from fz_openqa.datamodules.builders.corpus import MedQaCorpusBuilder
from fz_openqa.datamodules.index import ElasticSearchIndex, DenseIndex, Index
from fz_openqa.datamodules.index.index_pipes import FetchNestedDocuments
from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.inference.checkpoint import CheckpointLoader
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.functional import infer_batch_size
from fz_openqa.utils.pretty import get_separator
from fz_openqa.utils.pretty import pprint_batch

logger = logging.getLogger(__name__)

DEFAULT_CKPT = "https://drive.google.com/file/d/17XDASu1JYGndCFWNW3zCZGKD2woJuDIg/view?usp=sharing"
# define the default cache location
default_cache_dir = Path(fz_openqa.__file__).parent.parent / "cache"

OmegaConf.register_new_resolver("whoami", lambda: os.environ.get("USER"))
OmegaConf.register_new_resolver("getcwd", os.getcwd)


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
    CUDA_VISIBLE_DEVICES=0,1,2,3 poetry run python examples/index_corpus_with_faiss.py \
    trainer.strategy=dp trainer.gpus=4 +batch_size=1000 +num_workers=16 \
    +n_samples=1000 +use_subset=False +num_proc=4 +colbert=True

    ```
    """
    # set the context
    datasets.set_caching_enabled(True)
    datasets.logging.set_verbosity(datasets.logging.CRITICAL)
    transformers.logging.set_verbosity(transformers.logging.CRITICAL)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    seed_everything(1, workers=True)
    cache_dir = config.get("sys.cache_dir", default_cache_dir)
    use_colbert = config.get("colbert", False)

    # load model
    zero_shot = config.get("zero_shot", True)
    if zero_shot:
        bert_id = config.get("bert", "google/bert_uncased_L-2_H-128_A-2")
        model = ZeroShot(
            bert_id=bert_id,
            head="contextual" if use_colbert else "flat",
            limit_size=32 if use_colbert else None,
        )
        tokenizer = init_pretrained_tokenizer(pretrained_model_name_or_path=bert_id)
    else:
        loader = CheckpointLoader(config.get("checkpoint", DEFAULT_CKPT), override=config)
        if config.get("verbose", False):
            loader.print_config()
        model = loader.load_model(last=config.get("last", False))
        tokenizer = loader.tokenizer
    model.eval()
    model.freeze()
    logger.info(f"IndentityModel {type(model)} loaded")

    # Init Lightning trainer
    logger.info(f"Instantiating trainer <{config.trainer.get('_target_', None)}>")
    trainer: Optional[Trainer] = instantiate(config.trainer)
    if isinstance(trainer, (dict, DictConfig)):
        logger.info("No Trainer was provided. PyTorch Lightning acceleration cannot be used.")
        trainer = None

    # set up the corpus builder
    logger.info(f"Initialize corpus <{MedQaCorpusBuilder.__name__}>")
    corpus_builder = MedQaCorpusBuilder(
        tokenizer=tokenizer,
        to_sentences=config.get("to_sentences", False),
        use_subset=config.get("use_subset", True),
        add_encoding_tokens=not zero_shot,
        cache_dir=cache_dir,
        num_proc=config.get("num_proc", 2),
    )

    # build the corpus and take a subset
    corpus = corpus_builder()
    collate_fn = corpus_builder._get_collate_pipe()
    n_samples = config.get("n_samples", 1000)
    if n_samples is not None and n_samples > 0:
        n_samples = min(n_samples, len(corpus))
        corpus = corpus.select(range(n_samples))
    rich.print(corpus)

    # init the index
    IndexCls = ColbertIndex if use_colbert else DenseIndex
    if config.get("use_es", False):
        IndexCls = ElasticSearchIndex
    logger.info(f"Initialize index <{IndexCls.__name__}>")
    index: Index = IndexCls(
        dataset=corpus,
        model=model,
        trainer=trainer,
        faiss_train_size=config.get("faiss_train_size", 1000),
        # faiss_args={
        #     "factory": config.get("factory", "IVF100,PQ16x8"),
        #     "metric_type": faiss.METRIC_INNER_PRODUCT,
        # },
        loader_kwargs={
            "batch_size": config.get("batch_size", 10),
            "num_workers": config.get("num_workers", 1),
            "pin_memory": config.get("pin_memory", True),
        },
        model_output_keys=["_hd_", "_hq_"],
        collate_pipe=corpus_builder._get_collate_pipe(),
        cache_dir=cache_dir,
        persist_cache=config.get("persist_cache", False),
        in_memory=config.get("in_memory", True),
        dtype=config.get("dtype", "float32"),
        progress_bar=True,
        p=100,
    )
    rich.print(f"> index: {index}")

    # setup the fetch pipe (fetch all the other fields from the corpus)
    fetcher = FetchNestedDocuments(corpus_dataset=corpus_builder(), collate_pipe=collate_fn)
    query = collate_fn([corpus[i] for i in range(3)])
    query: Batch = {str(k).replace("document.", "question."): v for k, v in query.items()}

    # search for one batch
    pprint_batch(query, f"query {type(query)}")
    output = index(query, k=5)

    # format the output
    pprint_batch(output, "search result")
    output = {**query, **fetcher(output)}
    pprint_batch(output, "query + search results")
    stride = infer_batch_size(query)
    for i in range(min(stride, 3)):
        eg = Pipe.get_eg(output, idx=i)
        rich.print(get_separator())
        rich.print(f"query #{i + 1}: [cyan]{eg['question.text']}")
        for j in range(len(eg["document.text"])):
            rich.print(get_separator("."))
            txt = eg["document.text"][j]
            score = eg["document.proposal_score"][j]
            rich.print(f"doc #{j + 1}: score={score} [white]{txt}")

    # alternatively, you can search for a whole dataset:
    query_dset = corpus.rename_columns(
        {
            "document.text": "question.text",
            "document.input_ids": "question.input_ids",
            "document.attention_mask": "question.attention_mask",
        }
    )

    # search for the whole dataset
    question_collate = QaBuilder._get_collate_pipe(corpus_builder)
    query_dset = index(query_dset, k=3, collate_fn=question_collate, trainer=trainer)
    pprint_batch(query_dset[:3], "search whole dataset - results")


if __name__ == "__main__":
    run()
