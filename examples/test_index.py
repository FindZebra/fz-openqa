import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
from datasets import DatasetDict

from fz_openqa.datamodules.pipelines.collate.field import CollateField

sys.path.append(Path(__file__).parent.parent.as_posix())

from hydra.utils import instantiate
import os
from pathlib import Path

import torch
from torch import nn

from fz_openqa.datamodules.builders import QaBuilder, ConcatQaBuilder
from fz_openqa.datamodules.index.index import Index
import datasets
import hydra
import rich
from omegaconf import DictConfig

from fz_openqa import configs
from fz_openqa.datamodules.builders.corpus import CorpusBuilder
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer
from fz_openqa.utils.fingerprint import get_fingerprint
import loguru

# import the OmegaConf resolvers
from fz_openqa.training import experiment  # type: ignore


class RandnModel(nn.Module):
    def __init__(self, hdim: int = 32, dpr=True):
        super().__init__()
        self.hdim = hdim
        self.dummy = nn.Parameter(torch.zeros(1))
        self.dpr = nn.Parameter(int(dpr) + torch.zeros(1), requires_grad=False)

    def forward(self, x, **kwargs):
        output = {}
        if "document.vector" in x.keys():
            output["_hd_"] = x["document.vector"]
        elif "document.input_ids" in x.keys():
            output["_hd_"] = self._vector(x["document.input_ids"])
        if "question.vector" in x.keys():
            output["_hq_"] = x["question.vector"]
        elif "question.input_ids" in x.keys():
            output["_hq_"] = self._vector(x["question.input_ids"])
        return output

    def _vector(self, x):
        y = torch.randn(
            *x.shape,
            self.hdim,
            dtype=self.dummy.dtype,
            device=self.dummy.device,
        )
        if self.dpr > 0:
            y = y[..., 0, :]

        return y


@hydra.main(
    config_path=str(Path(configs.__file__).parent),
    config_name="script_config.yaml",
)
def run(config: DictConfig) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.multiprocessing.set_sharing_strategy("file_system")
    datasets.set_caching_enabled(True)
    # datasets.logging.set_verbosity(logging.ERROR)

    with tempfile.TemporaryDirectory(dir=config.sys.cache_dir) as tmpdir:
        # tmpdir = config.sys.cache_dir
        tmpdir = Path(tmpdir) / "test-index"
        tmpdir.mkdir(exist_ok=True, parents=True)

        # model
        model = RandnModel(dpr=False)
        rich.print(f"# [magenta] model : {get_fingerprint(model)}")

        # trainer
        trainer = instantiate(config.trainer)

        # initialize the tokenizer
        bert_id = "google/bert_uncased_L-2_H-128_A-2"
        tokenizer = init_pretrained_tokenizer(pretrained_model_name_or_path=bert_id)

        # initialize the corpus
        corpus_builder = CorpusBuilder(
            dset_name=config.get("dset_name", "medwiki"),
            tokenizer=tokenizer,
            use_subset=config.get("corpus_subset", True),
            cache_dir=config.sys.get("cache_dir"),
            num_proc=config.get("num_proc", 2),
            analytics=None,
            append_document_title=True,
        )
        corpus_builder.subset_size = [100]
        corpus = corpus_builder()
        document_ids = corpus["document.idx"]
        max_doc_id = max(document_ids)
        rich.print(f"> corpus: {corpus}, max_doc_id={max_doc_id}")

        # initialize the QA dataset
        qa_builder = ConcatQaBuilder(
            dset_name=config.get("dset_name", "medqa-us"),
            tokenizer=tokenizer,
            use_subset=config.get("use_subset", True),
            cache_dir=config.sys.get("cache_dir"),
            num_proc=config.get("num_proc", 2),
            analytics=None,
        )

        qa_dataset = qa_builder()
        rich.print(f"> qa_dataset: {qa_dataset}")

        # # add the document index columns
        # qa_dataset = DatasetDict(
        #     {
        #         split: dset.add_column(
        #             "question.document_idx",
        #             [np.random.randint(0, max_doc_id + 1) for _ in range(len(dset))],
        #         )
        #         for split, dset in qa_dataset.items()
        #     }
        # )

        # build the index
        index = Index(
            corpus,
            engines=[
                {
                    "name": "es",
                    "k": 100,
                    "merge_previous_results": True,
                    "max_batch_size": 100,
                    "verbose": False,
                    "config": {
                        "es_temperature": 10.0,
                        "auxiliary_weight": 2.0,
                    },
                },
                {
                    "name": "faiss_token",
                    "k": 1000,
                    "merge_previous_results": True,
                    "max_batch_size": 100,
                    # "verbose": True,
                    "config": {
                        "index_factory": "shard:IVF100000,PQ16",
                        "dimension": 32,
                        "p": 16,
                        "max_bs": 32,
                        "tempmem": 1 << 30,
                    },
                },
                {
                    "name": "maxsim",
                    "k": 100,
                    "merge_previous_results": False,
                    "max_batch_size": 100,
                    "config": {
                        "max_chunksize": 1000,
                    },
                },
            ],
            persist_cache=True,
            cache_dir=tmpdir,
            model=model,  # todo
            trainer=trainer,
            dtype="float16",
            corpus_collate_pipe=corpus_builder.get_collate_pipe(),
            dataset_collate_pipe=CollateField("question", tokenizer=tokenizer),
            loader_kwargs={
                "batch_size": 1000,
                "num_workers": config.get("num_proc", 2),
                "pin_memory": True,
            },
        )
        rich.print(f"> index: {index.fingerprint(reduce=True)}")

        # output = index(qa_dataset["train"][:3])
        # rich.print(output)

        rich.print("[green] ### Mapping the dataset")
        mapped_qa = index(
            qa_dataset,
            num_proc=config.get("num_proc", 2),
            batch_size=50,
            clean_caches=False,
            cache_fingerprint=tmpdir / "index_fingerprint",
        )
        rich.print(mapped_qa)

        rich.print("[magenta] ### Mapping the dataset (CACHE)")
        mapped_qa = index(
            qa_dataset,
            num_proc=config.get("num_proc", 2),
            batch_size=50,
            clean_caches=False,
            cache_fingerprint=tmpdir / "index_fingerprint",
        )
        rich.print(mapped_qa)

        # build the index
        # engine_config = {"es_temperature": 5.0}
        # es_engine_path = Path(tmpdir) / corpus._fingerprint
        # rich.print(f"[cyan]> es_engine_path: {es_engine_path}")
        # engine = AutoEngine("es", path=es_engine_path, config=engine_config, set_unique_path=True)
        # engine.build(corpus=corpus, vectors=None)
        # es_engine_path = engine.path
        # rich.print(f"> engine.base: {engine}")
        # del engine
        #
        # engine = IndexEngine.load_from_path(es_engine_path)
        # rich.print(f"> engine.loaded: {engine}")
        #
        # output = engine(qa_dataset["train"][:3])
        # rich.print(output)
        #
        # mapped_qa = engine(qa_dataset, set_new_fingerprint=True, num_proc=2)
        #
        # rich.print(mapped_qa)


if __name__ == "__main__":
    run()
