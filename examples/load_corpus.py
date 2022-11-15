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
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer
from loguru import logger

OmegaConf.register_new_resolver("whoami", lambda: os.environ.get("USER"))
OmegaConf.register_new_resolver("getcwd", os.getcwd)


@hydra.main(
    config_path=str(Path(configs.__file__).parent),
    config_name="script_config.yaml",
    version_base="1.2",
)
def run(config: DictConfig) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if config.get("disable_caching"):
        datasets.disable_caching()

    # initialize the tokenizer
    tokenizer = init_pretrained_tokenizer(
        pretrained_model_name_or_path=config.get("bert_id", "michiyasunaga/BioLinkBERT-base")
    )

    # initialize the data module
    builder = CorpusBuilder(
        dset_name=config.get("dset_name", "findzebra"),
        tokenizer=tokenizer,
        subset_size=config.get("subset_size", None),
        cache_dir=config.sys.get("cache_dir"),
        num_proc=config.get("num_proc", 2),
        add_qad_tokens=config.get("add_qad_tokens", False),
        append_document_title=config.get("append_title", True),
        passage_length=config.get("passage_length", 200),
        passage_stride=config.get("passage_stride", 100),
        analytics=[
            SequenceLengths(
                output_dir="analytics/",
                verbose=True,
                concatenate=config.get("concatenate_splits", False),
            ),
        ],
    )
    dm = DataModule(builder=builder, num_proc=config.get("num_proc", 0))

    # build the dataset
    dm.setup()
    dm.display_samples(n_samples=20)

    # access dataset
    rich.print(dm.dataset)
    logger.info(f"n. unique documents: {len(set(dm.dataset['document.idx']))}")

    # sample a batch
    _ = next(iter(dm.train_dataloader()))


if __name__ == "__main__":
    run()
