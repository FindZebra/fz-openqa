import logging
import os
from pathlib import Path

import datasets
import hydra
import rich
from omegaconf import DictConfig
from rich.logging import RichHandler

import fz_openqa
from fz_openqa import configs
from fz_openqa.datamodules.builders.pubmedqa import PubMedQaBuilder
from fz_openqa.datamodules.datamodule import DataModule
from fz_openqa.datamodules.utils.dataset import get_column_names
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

    # initialize the data module
    builder = PubMedQaBuilder(
        tokenizer=tokenizer,
        use_subset=config.get("use_subset", True),
        cache_dir=config.get("cache_dir", default_cache_dir),
        num_proc=2,
    )

    dm = DataModule(builder=builder)
    dm.prepare_data()
    dm.setup()
    dm.display_samples()  # throws error with invalid data type

    # access dataset
    rich.print(dm.dataset)


if __name__ == "__main__":
    run()
