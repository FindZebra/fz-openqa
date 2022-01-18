import cProfile
import logging
import pstats
from pathlib import Path
from timeit import Timer

import datasets
import hydra
import numpy as np
import rich
from rich.logging import RichHandler

import fz_openqa
from fz_openqa import configs
from fz_openqa.datamodules.__old.corpus_dm import MedQaCorpusDataModule
from fz_openqa.datamodules.__old.meqa_dm import MedQaDataModule
from fz_openqa.datamodules.builders.corpus import MedQaCorpusBuilder
from fz_openqa.datamodules.builders.openqa import OpenQaBuilder
from fz_openqa.datamodules.builders.qa import QaBuilder
from fz_openqa.datamodules.datamodule import DataModule
from fz_openqa.datamodules.index import ElasticSearchIndex
from fz_openqa.datamodules.index import ElasticSearchIndexBuilder
from fz_openqa.datamodules.pipes import ExactMatch
from fz_openqa.datamodules.pipes import TextFormatter
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer
from fz_openqa.utils.pretty import get_separator
from fz_openqa.utils.train_utils import setup_safe_env


@hydra.main(
    config_path=str(Path(configs.__file__).parent),
    config_name="script_config.yaml",
)
def run(config):
    datasets.set_caching_enabled(True)
    setup_safe_env()

    # define the default cache location
    default_cache_dir = Path(fz_openqa.__file__).parent.parent / "cache"

    tokenizer = init_pretrained_tokenizer(pretrained_model_name_or_path="bert-base-cased")

    text_formatter = TextFormatter(lowercase=True)

    # define the medqa builder
    dataset_builder = QaBuilder(
        tokenizer=tokenizer,
        text_formatter=text_formatter,
        use_subset=config.get("use_subset", True),
        cache_dir=config.get("cache_dir", default_cache_dir),
        num_proc=4,
    )
    dataset_builder.subset_size = [1000, 100, 100]

    # define the corpus builder
    corpus_builder = MedQaCorpusBuilder(
        tokenizer=tokenizer,
        text_formatter=text_formatter,
        use_subset=False,
        cache_dir=config.get("cache_dir", default_cache_dir),
        num_proc=4,
    )

    # define the OpenQA builder
    builder = OpenQaBuilder(
        dataset_builder=dataset_builder,
        corpus_builder=corpus_builder,
        index_builder=ElasticSearchIndexBuilder(),
        relevance_classifier=ExactMatch(),
        n_retrieved_documents=1000,
        n_documents=1000,
        max_pos_docs=None,
        filter_unmatched=True,
        num_proc=2,
        batch_size=10,
    )

    # define the data module
    dm = DataModule(builder=builder, train_batch_size=10)

    # prepare both the QA dataset and the corpus
    dm.prepare_data()
    dm.setup()
    rich.print(dm.dataset)

    class GetBatch:
        def __init__(self, loader):
            self.it_loader = iter(loader)

        def __call__(self):
            return next(self.it_loader)

    get_batch = GetBatch(dm.train_dataloader())

    profiler = cProfile.Profile()
    profiler.enable()
    times = Timer(get_batch).repeat(10, 1)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("time")
    stats.print_stats(20)
    print(get_separator())
    rich.print(f">> duration={np.mean(times):.3f}s/batch (std={np.std(times):.3f}s)")
    # A. < 31-10-2021 - testing fetching in __getitem__ vs. in dataloader
    # fetch docs in collate: >> duration=0.142s/batch (std=0.293s)
    # fetch docs in __getitem__: >> duration=1.351s/batch (std=2.645s)make

    # B. fetching without loading the whole column (1000 docs)
    # index: >> duration=1.952s/batch (std=1.935s)
    # hf_take: >> duration=1.380s/batch (std=1.250s)

    # C. fetching without loading the whole column (100 docs)
    # index: >> duration=0.246s/batch (std=0.250s)
    # hf_take: >> duration=0.194s/batch (std=0.197s)

    # D. fetching without loading the whole column (10 docs)
    # index: >> duration=0.090s/batch (std=0.105s)
    # hf_take: >> duration=0.088s/batch (std=0.102s)

    # E. Final - without formatting
    # 10 docs: >> duration=0.094s/batch (std=0.107s)
    # 100 docs: >> duration=0.228s/batch (std=0.226s)
    # 1000 docs: >> duration=1.722s/batch (std=1.635s)

    # E. Final
    # 10 docs: >> duration=0.097s/batch (std=0.111s)
    # 100 docs: >> duration=0.304s/batch (std=0.298s)
    # 1000 docs: >> duration=2.529s/batch (std=2.401s)


if __name__ == "__main__":
    run()
