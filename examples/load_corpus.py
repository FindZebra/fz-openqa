import logging
import os
import sys
from pathlib import Path

from fz_openqa.datamodules.analytics import SequenceLengths

sys.path.append(Path(__file__).parent.parent.as_posix())

import datasets
import hydra
import rich
from omegaconf import DictConfig
from omegaconf import OmegaConf

from fz_openqa import configs
from fz_openqa.datamodules.builders.corpus import CorpusBuilder
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
    tokenizer = init_pretrained_tokenizer(
        pretrained_model_name_or_path="michiyasunaga/BioLinkBERT-base"
    )

    # initialize text formatter
    textformatter = TextFormatter(
        lowercase=True,
        remove_symbols=True,
        remove_ref=True,
        remove_hex=True,
        remove_breaks=True,
    )

    # initialize the data module
    builder = CorpusBuilder(
        dset_name=config.get("dset_name", "medqa"),
        tokenizer=tokenizer,
        to_sentences=config.get("to_sentences", False),
        text_formatter=textformatter,
        subset_size=config.get("subset_size", None),
        cache_dir=config.sys.get("cache_dir"),
        num_proc=config.get("num_proc", 2),
        append_document_title=config.get("append_title", True),
        passage_length=config.get("window_size", 200),
        passage_stride=config.get("window_stride", 100),
        analytics=[
            SequenceLengths(
                output_dir="analytics/",
                verbose=True,
                concatenate=config.get("concatenate_splits", False),
            ),
        ],
    )
    dm = DataModule(builder=builder)

    dm.setup()
    dm.display_samples(n_samples=5)

    # access dataset
    rich.print(dm.dataset)
    rich.print(f"> n documents: {len(set(dm.dataset['document.idx']))}")

    # sample a batch
    _ = next(iter(dm.train_dataloader()))


if __name__ == "__main__":
    run()
