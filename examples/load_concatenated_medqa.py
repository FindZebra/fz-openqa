import logging
import os
from pathlib import Path

import datasets
import hydra
import rich
from omegaconf import DictConfig

import fz_openqa
from fz_openqa import configs
from fz_openqa.datamodules.builders.qa import ConcatQaBuilder
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

    # define the default cache location
    default_cache_dir = Path(fz_openqa.__file__).parent.parent / "cache"

    # initialize the tokenizer
    tokenizer = init_pretrained_tokenizer(pretrained_model_name_or_path="bert-base-cased")

    # initialize the data module
    builder = ConcatQaBuilder(
        tokenizer=tokenizer,
        use_subset=config.get("use_subset", False),
        cache_dir=config.get("cache_dir", default_cache_dir),
        query_expansion=config.get("query_expansion", None),
        num_proc=config.get("num_proc", 2),
        dset_name=config.get("dset_name", "medqa-us"),
    )
    dm = DataModule(builder=builder)
    dm.prepare_data()
    dm.setup()
    dm.display_samples(n_samples=3)

    # access dataset
    rich.print(dm.dataset)

    # sample a batch
    batch = next(iter(dm.train_dataloader()))
    rich.print(batch["question.input_ids"][0])
    rich.print(batch["question.attention_mask"][0])


if __name__ == "__main__":
    run()
