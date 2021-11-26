import logging
import os
from pathlib import Path

import datasets
import hydra
import rich
from omegaconf import DictConfig

import fz_openqa
from fz_openqa import configs
from fz_openqa.datamodules.analytics.corpus_statistics import ReportCorpusStatistics
from fz_openqa.datamodules.builders.corpus import MedQaCorpusBuilder, MedWikipediaCorpusBuilder, FzCorpusBuilder
from fz_openqa.datamodules.builders.medqa_x_wikipedia_corpus import WikixMedQaCorpusBuilder
from fz_openqa.datamodules.datamodule import DataModule
from fz_openqa.datamodules.pipes import TextFormatter
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer

logger = logging.getLogger(__name__)


@hydra.main(
    config_path=str(Path(configs.__file__).parent),
    config_name="script_config.yaml",
)
def run(config: DictConfig) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    datasets.set_caching_enabled(True)

    # define the default cache location
    default_cache_dir = Path(fz_openqa.__file__).parent.parent / "cache"

    # initialize the tokenizer
    tokenizer = init_pretrained_tokenizer(pretrained_model_name_or_path="bert-base-cased")

    # initialize text formatter
    textformatter = TextFormatter(
        lowercase=True,
        remove_symbols=True,
        remove_ref=True,
        remove_hex=True,
        remove_breaks=True,
    )

    # initialize the data module
    builder = MedWikipediaCorpusBuilder(
        tokenizer=tokenizer,
        to_sentences=config.get("to_sentences", True),
        text_formatter=textformatter,
        use_subset=config.get("use_subset", False),
        cache_dir=config.get("cache_dir", default_cache_dir),
        num_proc=2,
        analyses=[ReportCorpusStatistics(output_dir="./analyses")]
    )
    dm = DataModule(builder=builder)
    dm.prepare_data()
    dm.setup()
    dm.display_samples(n_samples=5)

    # access dataset
    rich.print(dm.dataset)

    # sample a batch
    _ = next(iter(dm.train_dataloader()))


if __name__ == "__main__":
    run()
