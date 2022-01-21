import logging
import os
from pathlib import Path

import datasets
import hydra
import rich
from omegaconf import DictConfig
from rich.logging import RichHandler

from fz_openqa import configs
from fz_openqa.datamodules.builders.qa import QaBuilder
from fz_openqa.datamodules.datamodule import DataModule
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer

logger = logging.getLogger(__name__)


@hydra.main(
    config_path=str(Path(configs.__file__).parent),
    config_name="script_config.yaml",
)
def run(config: DictConfig) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    datasets.set_caching_enabled(True)

    # initialize the tokenizer
    tokenizer = init_pretrained_tokenizer(pretrained_model_name_or_path="bert-base-cased")

    # initialize the data module
    builder = QaBuilder(
        tokenizer=tokenizer,
        use_subset=config.get("use_subset", False),
        cache_dir=config.sys.cache_dir,
        question_length=config.get("question_length", None),
        num_proc=config.get("num_proc", 2),
        dset_name=config.get("dset_name", "medqa-us"),
    )
    dm = DataModule(builder=builder)
    dm.prepare_data()
    dm.setup()
    dm.display_samples(n_samples=3)

    # access dataset
    rich.print(dm.dataset)


if __name__ == "__main__":
    run()