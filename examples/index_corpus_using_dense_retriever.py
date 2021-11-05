import io
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional
from typing import Union

import datasets
import hydra
import rich
import torch
import xxhash
from hydra._internal.instantiate._instantiate2 import _resolve_target
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

import fz_openqa
from fz_openqa import configs
from fz_openqa.datamodules.builders.corpus import MedQaCorpusBuilder
from fz_openqa.datamodules.index import FaissIndex
from fz_openqa.datamodules.index.dense import fingerprint_model
from fz_openqa.datamodules.pipes import ApplyAsFlatten
from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.datamodules.pipes import SearchCorpus
from fz_openqa.datamodules.pipes import Sort
from fz_openqa.datamodules.pipes.control.filter_keys import KeyIn
from fz_openqa.datamodules.pipes.control.filter_keys import KeyWithPrefix
from fz_openqa.datamodules.pipes.nesting import infer_batch_size
from fz_openqa.datamodules.pipes.search import FetchDocuments
from fz_openqa.modeling import Model
from fz_openqa.tokenizers.pretrained import PreTrainedTokenizerFast
from fz_openqa.utils.config import print_config
from fz_openqa.utils.pretty import get_separator
from fz_openqa.utils.pretty import pprint_batch

logger = logging.getLogger(__name__)

DEFAULT_CKPT = "/Users/valv/Documents/Research/remote_data/12-35-00"
# define the default cache location
default_cache_dir = Path(fz_openqa.__file__).parent.parent / "cache"

OmegaConf.register_new_resolver("whoami", lambda: os.environ.get("USER"))
OmegaConf.register_new_resolver("getcwd", os.getcwd)


class CheckpointLoader:
    _tokenizer: Optional[PreTrainedTokenizerFast] = None

    def __init__(self, checkpoint_dir: str, override=Optional[OmegaConf]):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.config = OmegaConf.load(self.checkpoint_dir / "config.yaml")
        if override is not None:
            self.config.update(override)

    def pprint(self):
        print_config(self.config, resolve=False)

    def model_checkpoint(self, last=False) -> Union[None, Path]:
        checkpoints = (self.checkpoint_dir / "checkpoints").iterdir()
        checkpoints = filter(lambda x: x.suffix == ".ckpt", checkpoints)
        if last:
            checkpoints = filter(lambda x: "last" in x.name, checkpoints)
        else:
            checkpoints = filter(lambda x: "last" not in x.name, checkpoints)

        try:
            return next(iter(checkpoints))
        except StopIteration:
            return None

    def load_tokenizer(self):
        return instantiate(self.config.datamodule.tokenizer)

    @property
    def tokenizer(self) -> PreTrainedTokenizerFast:
        if self._tokenizer is None:
            self._tokenizer = self.load_tokenizer()
        return self._tokenizer

    def load_bert(self):
        return instantiate(self.config.model.bert)

    def load_model(self, last=False) -> Model:

        logger.info(f"Instantiating model <{self.config.model._target_}>")

        path = self.model_checkpoint(last=last)
        if path is not None:
            logger.info(f"Loading model from checkpoint: {path}")
            # need to override the saved `tokenizer` and `bert` hyperparameters
            # so `sys.cache_dir` can be overridden
            cls = _resolve_target(self.config.model._target_)
            model = cls.load_from_checkpoint(
                path,
                tokenizer=OmegaConf.to_object(self.config.datamodule.tokenizer),
                bert=OmegaConf.to_object(self.config.model.bert),
            )
        else:
            logger.warning("No checkpoint found. Initializing model without checkpoint.")
            model = instantiate(self.config.model, _recursive_=False)

        return model


@torch.no_grad()
@hydra.main(
    config_path=str(Path(configs.__file__).parent),
    config_name="script_config.yaml",
)
def run(config: DictConfig) -> None:
    seed_everything(1, workers=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    datasets.set_caching_enabled(True)
    loader = CheckpointLoader(config.get("checkpoint", DEFAULT_CKPT), override=config)
    # loader.pprint()
    model = loader.load_model(last=config.get("last", False))
    # todo: check if model is properly loaded

    raw_bert = loader.load_bert()
    check_bert(model.module.bert, "model.module.bert")
    check_bert(raw_bert, "load_bert")

    logger.info(f"Initialize corpus <{MedQaCorpusBuilder.__name__}>")
    corpus_builder = MedQaCorpusBuilder(
        tokenizer=loader.tokenizer,
        to_sentences=config.get("to_sentences", False),
        use_subset=config.get("use_subset", True),
        cache_dir=config.get("sys.cache_dir", default_cache_dir),
        num_proc=2,
    )
    corpus = corpus_builder().select(range(100))
    rich.print(corpus)

    logger.info(f"Initialize index <{FaissIndex.__name__}>")
    rich.print(f">> model.fingerprint={fingerprint_model(model.module.bert)}")
    index = FaissIndex(
        dataset=corpus,
        model=model,
        batch_size=10,
        model_output_keys=["_hd_", "_hq_"],
        cache_dir=tempfile.tempdir,
    )
    rich.print(index)

    # setup search pipe
    search = SearchCorpus(index, k=3, model=model)
    sorter = ApplyAsFlatten(
        Sort(keys=["document.retrieval_score"]),
        filter=KeyWithPrefix("document."),
    )
    fetcher = ApplyAsFlatten(
        FetchDocuments(
            corpus_dataset=corpus_builder(),
            collate_pipe=corpus_builder.get_collate_pipe(),
        ),
        filter=KeyIn(["document.row_idx"]),
        update=True,
    )

    # query
    query = corpus[:2]
    query = {str(k).replace("document.", "question."): v for k, v in query.items()}
    output = sorter(search(query))
    pprint_batch(output, "query output")
    output = {**query, **fetcher(output)}
    pprint_batch(output, "query output + all fields")
    for i in range(infer_batch_size(query)):
        eg = Pipe.get_eg(output, idx=i)
        rich.print(get_separator())
        rich.print(f"query: [white]{eg['question.text']}")
        for j in range(len(eg["document.text"])):
            rich.print(get_separator("."))
            txt = eg["document.text"][j]
            score = eg["document.retrieval_score"][j]
            rich.print(f"doc: score={score} [white]{txt}")


def check_bert(bert, header=""):
    params = list(sorted(bert.named_parameters(), key=lambda x: x[0]))
    rich.print(f"[green]=== {header} : n_params={len(params)} :hash={fingerprint_model(bert)} ===")


if __name__ == "__main__":
    run()
