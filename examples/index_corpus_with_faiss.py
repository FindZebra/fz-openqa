import os
import sys

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
from utils import ZeroShot
from fz_openqa import configs
from fz_openqa.callbacks.store_results import StorePredictionsCallback
from fz_openqa.datamodules.builders.corpus import MedQaCorpusBuilder
from fz_openqa.datamodules.index import FaissIndex
from fz_openqa.datamodules.index.colbert import ColbertIndex
from fz_openqa.datamodules.index.pipes import FetchNestedDocuments
from fz_openqa.datamodules.index.pipes import SearchCorpus
from fz_openqa.datamodules.pipes import Pipe
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
    zero_shot = config.get("zero_shot", True)
    if zero_shot:
        bert_id = config.get("bert", "google/bert_uncased_L-2_H-128_A-2")
        model = ZeroShot(bert_id=bert_id)
        tokenizer = init_pretrained_tokenizer(pretrained_model_name_or_path=bert_id)
    else:
        loader = CheckpointLoader(config.get("checkpoint", DEFAULT_CKPT), override=config)
        if config.get("verbose", False):
            loader.print_config()
        model = loader.load_model(last=config.get("last", False))
        tokenizer = loader.tokenizer
    model.eval()
    model.freeze()
    logger.info(f"Model {type(model)} loaded")

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
    collate_fn = corpus_builder.get_collate_pipe()
    n_samples = config.get("n_samples", 1000)
    if n_samples is not None and n_samples > 0:
        n_samples = min(n_samples, len(corpus))
        corpus = corpus.select(range(n_samples))
    rich.print(corpus)

    # init the index
    logger.info(f"Initialize index <{ColbertIndex.__name__}>")
    index = ColbertIndex(
        dataset=corpus,
        model=model,
        trainer=trainer,
        faiss_args={"metric_type": faiss.METRIC_L2, "n_list": 8, "n_subvectors": 8, "n_bits": 8},
        loader_kwargs={
            "batch_size": config.get("batch_size", 10),
            "num_workers": config.get("num_workers", 1),
            "pin_memory": config.get("pin_memory", True),
        },
        model_output_keys=["_hd_", "_hq_"],
        collate_pipe=corpus_builder.get_collate_pipe(),
        cache_dir=cache_dir,
    )
    rich.print(index.is_indexed)
    rich.print(index.ntotal)

    # setup search pipe (query the indexes from the corpus)
    search = SearchCorpus(index, k=3)
    # setup the fetch pipe (fetch all the other fields from the corpus)
    fetcher = FetchNestedDocuments(corpus_dataset=corpus_builder(), collate_pipe=collate_fn)
    query = collate_fn([corpus[i] for i in range(3)])
    query: Batch = {str(k).replace("document.", "question."): v for k, v in query.items()}

    # search for one batch
    pprint_batch(query, "query")
    output = search(query)

    # format the output
    pprint_batch(output, "search result")
    output = {**query, **fetcher(output)}
    pprint_batch(output, "query + search results")
    for i in range(infer_batch_size(query)):
        eg = Pipe.get_eg(output, idx=i)
        rich.print(get_separator())
        rich.print(f"query #{i + 1}: [cyan]{eg['question.text']}")
        for j in range(len(eg["document.text"])):
            rich.print(get_separator("."))
            txt = eg["document.text"][j]
            score = eg["document.retrieval_score"][j]
            rich.print(f"doc #{j + 1}: score={score} [white]{txt}")

    # alternatively, you can iterate you can search for a whole dataloader:
    outputs = index.search(corpus, k=3, collate_fn=collate_fn)
    for search_results in outputs:
        pprint_batch(vars(search_results), "search whole dataset - results")
        break


if __name__ == "__main__":
    run()
