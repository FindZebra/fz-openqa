import logging
import os
from pathlib import Path
from typing import Optional

import datasets
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
from fz_openqa import configs
from fz_openqa.datamodules import DataModule
from fz_openqa.datamodules.builders import MedQaBuilder
from fz_openqa.datamodules.builders import OpenQaBuilder
from fz_openqa.datamodules.builders.corpus import MedQaCorpusBuilder
from fz_openqa.datamodules.index import FaissIndexBuilder
from fz_openqa.datamodules.pipes import ExactMatch
from fz_openqa.inference.checkpoint import CheckpointLoader
from fz_openqa.utils.config import print_config
from fz_openqa.utils.fingerprint import get_fingerprint

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
    # set the context
    datasets.set_caching_enabled(True)
    datasets.logging.set_verbosity(datasets.logging.CRITICAL)
    transformers.logging.set_verbosity(transformers.logging.CRITICAL)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    seed_everything(1, workers=True)
    cache_dir = config.get("sys.cache_dir", default_cache_dir)
    print_config(config)

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
    trainer: Optional[Trainer] = instantiate(
        config.trainer,
    )
    if isinstance(trainer, (dict, DictConfig)):
        logger.info("No Trainer was provided. PyTorch Lightning acceleration cannot be used.")
        trainer = None

    # set up the corpus builder
    logger.info(f"Initialize corpus <{MedQaCorpusBuilder.__name__}>")
    corpus_builder = MedQaCorpusBuilder(
        tokenizer=loader.tokenizer,
        to_sentences=config.get("to_sentences", False),
        use_subset=config.get("corpus_subset", True),
        cache_dir=cache_dir,
        num_proc=config.get("num_proc", 2),
    )
    corpus_builder.subset_size = [
        1,
    ]

    # setup the dataset builder
    logger.info(f"Initialize dataset <{MedQaBuilder.__name__}>")
    dataset_builder = MedQaBuilder(
        tokenizer=loader.tokenizer,
        use_subset=config.get("use_subset", True),
        cache_dir=cache_dir,
        num_proc=config.get("num_proc", 2),
    )
    dataset_builder.subset_size = [100, 100, 100]

    # init the index builder
    logger.info(f"Initialize index <{FaissIndexBuilder.__name__}>")
    index_builder = FaissIndexBuilder(
        model=model,
        trainer=trainer,
        loader_kwargs={
            "batch_size": config.get("batch_size", 10),
            "num_workers": config.get("num_workers", 4),
            "pin_memory": config.get("pin_memory", True),
        },
        model_output_keys=["_hd_", "_hq_"],
        collate_pipe=corpus_builder.get_collate_pipe(),
        persist_cache=config.get("persist_cache", True),
        cache_dir=cache_dir,
    )

    # define the OpenQA builder
    logger.info(f"Initialize builder <{OpenQaBuilder.__name__}>")
    builder = OpenQaBuilder(
        dataset_builder=dataset_builder,
        corpus_builder=corpus_builder,
        index_builder=index_builder,
        relevance_classifier=ExactMatch(interpretable=False),
        n_retrieved_documents=100,
        n_documents=None,
        max_pos_docs=10,
        filter_unmatched=config.get("filter_unmatched", True),
        num_proc=config.get("num_proc", 2),
        batch_size=config.get("batch_size", 2),
    )

    # define the data module
    dm = DataModule(builder=builder)

    # preprocess the data
    dm.prepare_data()
    dm.setup()
    dm.display_samples(n_samples=3)

    rich.print(f"> dm.dataset.index={dm.dataset.index}")
    exit()

    # setup search pipe (query the indexes from the corpus)
    # search = SearchCorpus(dm.dataset.index, k=3, model=model)
    # # setup the fetch pipe (fetch all the other fields from the corpus)
    # fetcher = FetchNestedDocuments(
    #     corpus_dataset=corpus_builder(), collate_pipe=corpus_builder.get_collate_pipe()
    # )
    #
    # # get the query from the corpus
    # query = corpus_builder.get_collate_pipe()([corpus[i] for i in range(3)])
    #
    # # search the index (for this example, we use the document.input_ids
    # # to make sure that the same BERT head is used)
    # # however the field is renamed afterwards, for display purposes
    # output = search(query)
    # query: Batch = {str(k).replace("document.", "question."): v for k, v in query.items()}
    #
    # # format the output
    # pprint_batch(output, "query output")
    # output = {**query, **fetcher(output)}
    # pprint_batch(output, "query output + all fields")
    # for i in range(infer_batch_size(query)):
    #     eg = Pipe.get_eg(output, idx=i)
    #     rich.print(get_separator())
    #     rich.print(f"query #{i + 1}: [cyan]{eg['question.text']}")
    #     for j in range(len(eg["document.text"])):
    #         rich.print(get_separator("."))
    #         txt = eg["document.text"][j]
    #         score = eg["document.retrieval_score"][j]
    #         rich.print(f"doc #{j + 1}: score={score} [white]{txt}")


if __name__ == "__main__":
    run()
