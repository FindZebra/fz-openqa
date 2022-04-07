import logging
import os
import sys
from pathlib import Path

sys.path.append(Path(__file__).parent.parent.as_posix())

import datasets
import hydra
import rich
from omegaconf import DictConfig
from omegaconf import OmegaConf

from fz_openqa import configs
from fz_openqa.datamodules.analytics.corpus_statistics import ReportCorpusStatistics
from fz_openqa.datamodules.builders.corpus import FzCorpusBuilder
from fz_openqa.datamodules.builders.corpus import MedQaCorpusBuilder
from fz_openqa.datamodules.builders.corpus import MedWikipediaCorpusBuilder
from fz_openqa.datamodules.builders.corpus import QuALITYCorpusBuilder

try:
    from fz_openqa.datamodules.builders.medqa_x_wikipedia_corpus import WikixMedQaCorpusBuilder
except ImportError:
    WikixMedQaCorpusBuilder = None
from fz_openqa.datamodules.datamodule import DataModule
from fz_openqa.datamodules.pipes import TextFormatter
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer

logger = logging.getLogger(__name__)

OmegaConf.register_new_resolver("whoami", lambda: os.environ.get("USER"))
OmegaConf.register_new_resolver("getcwd", os.getcwd)


@hydra.main(
    config_path=str(Path(configs.__file__).parent),
    config_name="script_config.yaml",
)
def run(config: DictConfig) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    datasets.set_caching_enabled(True)

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

    Cls = {
        "medqa": MedQaCorpusBuilder,
        "fz": FzCorpusBuilder,
        "medwikipedia": MedWikipediaCorpusBuilder,
        "wikixmedqa": WikixMedQaCorpusBuilder,
        "quality": QuALITYCorpusBuilder,
    }[config.get("corpus", "medwikipedia")]

    # initialize the data module
    builder = Cls(
        tokenizer=tokenizer,
        to_sentences=config.get("to_sentences", False),
        text_formatter=textformatter,
        use_subset=config.get("use_subset", True),
        cache_dir=config.sys.get("cache_dir"),
        num_proc=2,
        analytics=[ReportCorpusStatistics(verbose=True, output_dir="./analyses")],
        append_document_title=config.get("append_title", True),
    )
    dm = DataModule(builder=builder)

    dm.setup()
    dm.display_samples(n_samples=5)

    # access dataset
    rich.print(dm.dataset)

    # sample a batch
    _ = next(iter(dm.train_dataloader()))


if __name__ == "__main__":
    run()
