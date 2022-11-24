import os
import sys
from pathlib import Path

sys.path.append(Path(__file__).parent.parent.as_posix())

import datasets
import hydra
import rich
from omegaconf import DictConfig, OmegaConf

from fz_openqa import configs
from fz_openqa.datamodules.builders.qa import QaBuilder
from fz_openqa.datamodules.datamodule import DataModule
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer
from fz_openqa.datamodules.analytics import SequenceLengths
from fz_openqa.datamodules.builders.preprocessing.entity import EntityPreprocessing


OmegaConf.register_new_resolver("whoami", lambda: os.environ.get("USER"))
OmegaConf.register_new_resolver("getcwd", os.getcwd)


@hydra.main(
    config_path=str(Path(configs.__file__).parent),
    config_name="script_config.yaml",
)
def run(config: DictConfig) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if config.get("disable_caching"):
        datasets.disable_caching()

    # initialize the tokenizer
    bert_id = config.get("bert_id", "michiyasunaga/BioLinkBERT-base")
    tokenizer = init_pretrained_tokenizer(pretrained_model_name_or_path=bert_id)

    # preprocessing
    preprocessing_op = config.get("preprocessing", None)
    if isinstance(preprocessing_op, str):
        preprocessing_op = {"entity": EntityPreprocessing}[preprocessing_op]()

    if config.get("use_subset", False):
        subset_size = 100
    else:
        subset_size = None

    # initialize the data module
    builder = QaBuilder(
        tokenizer=tokenizer,
        subset_size=subset_size,
        cache_dir=config.sys.cache_dir,
        query_expansion=config.get("query_expansion", None),
        concat_qa=config.get("concat_qa", False),
        n_answer_tokens=0,
        num_proc=config.get("num_proc", 2),
        dset_name=config.get("dset_name", "medqa-us"),
        max_length=config.get("max_length", None),
        preprocessing_op=preprocessing_op,
        analytics=[
            SequenceLengths(
                output_dir="analytics/",
                verbose=True,
                concatenate=config.get("concatenate_splits", False),
            ),
        ],
    )
    dm = DataModule(builder=builder, num_workers=config.get("num_workers", 0))

    # build the dataset
    dm.setup()
    dm.display_samples(n_samples=1)

    # access dataset
    rich.print(dm.dataset)

    # sample a batch
    _ = next(iter(dm.train_dataloader()))


if __name__ == "__main__":
    run()
