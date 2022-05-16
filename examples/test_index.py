import logging
import sys
import tempfile
from pathlib import Path


sys.path.append(Path(__file__).parent.parent.as_posix())

from hydra.utils import instantiate
import os
from pathlib import Path

import torch
from torch import nn

from fz_openqa.datamodules.builders import QaBuilder
from fz_openqa.datamodules.index.base import Index
from fz_openqa.datamodules.index.engines.es import ElasticsearchEngine
from fz_openqa.datamodules.index.engines.base import IndexEngine
from fz_openqa.utils.fingerprint import get_fingerprint
from fz_openqa.datamodules.index.engines.auto import AutoEngine
import datasets
import hydra
import rich
from omegaconf import DictConfig

from fz_openqa import configs
from fz_openqa.datamodules.builders.corpus import CorpusBuilder
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer

# import te omegaconf resolvers
from fz_openqa.training import experiment  # type: ignore


class IndentityModel(nn.Module):
    def __init__(self, hdim: int = 32):
        super().__init__()
        self.hdim = hdim
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x, **kwargs):
        output = {}
        if "document.vector" in x.keys():
            output["_hd_"] = x["document.vector"]
        elif "document.input_ids" in x.keys():
            output["_hd_"] = torch.randn(
                *x["document.input_ids"].shape,
                self.hdim,
                dtype=self.dummy.dtype,
                device=self.dummy.device,
            )
        if "question.vector" in x.keys():
            output["_hq_"] = x["question.vector"]
        elif "question.input_ids" in x.keys():
            output["_hq_"] = torch.randn(
                *x["question.input_ids"].shape,
                self.hdim,
                dtype=self.dummy.dtype,
                device=self.dummy.device,
            )
        return output


@hydra.main(
    config_path=str(Path(configs.__file__).parent),
    config_name="script_config.yaml",
)
def run(config: DictConfig) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    datasets.set_caching_enabled(True)
    datasets.logging.set_verbosity(logging.ERROR)

    with tempfile.TemporaryDirectory(dir=config.sys.cache_dir) as tmpdir:
        tmpdir = Path(config.sys.cache_dir) / "test-index"
        tmpdir.mkdir(exist_ok=True, parents=True)

        # model
        # model = IndentityModel()

        # trainer
        # trainer = instantiate(config.trainer)

        # initialize the tokenizer
        tokenizer = init_pretrained_tokenizer(pretrained_model_name_or_path="bert-base-cased")

        # initialize the corpus
        corpus_builder = CorpusBuilder(
            dset_name=config.get("dset_name", "medqa"),
            tokenizer=tokenizer,
            use_subset=config.get("corpus_subset", False),
            cache_dir=config.sys.get("cache_dir"),
            num_proc=config.get("num_proc", 2),
            analytics=None,
            append_document_title=True,
        )
        corpus = corpus_builder()
        rich.print(f"> corpus: {corpus}")

        # initialize the QA dataset
        qa_builder = QaBuilder(
            dset_name=config.get("dset_name", "medqa-us"),
            tokenizer=tokenizer,
            use_subset=config.get("use_subset", True),
            cache_dir=config.sys.get("cache_dir"),
            num_proc=config.get("num_proc", 2),
            analytics=None,
        )
        qa_dataset = qa_builder()
        rich.print(f"> qa_dataset: {qa_dataset}")

        # build the index
        engine_config = {"es_temperature": 5.0}
        es_engine_path = Path(tmpdir) / corpus._fingerprint
        rich.print(f"[cyan]> es_engine_path: {es_engine_path}")
        engine = AutoEngine("es", path=es_engine_path, config=engine_config, set_unique_path=True)
        engine.build(corpus=corpus, vectors=None)
        es_engine_path = engine.path
        rich.print(f"> engine.base: {engine}")
        del engine

        engine = IndexEngine.load_from_path(es_engine_path)
        rich.print(f"> engine.loaded: {engine}")

        output = engine(qa_dataset["train"][:3])
        rich.print(output)

        mapped_qa = engine(qa_dataset, set_new_fingerprint=True, num_proc=2)

        rich.print(mapped_qa)


if __name__ == "__main__":
    run()
