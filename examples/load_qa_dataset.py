import os
import sys
from pathlib import Path

sys.path.append(Path(__file__).parent.parent.as_posix())

import datasets
import hydra
import rich
from omegaconf import DictConfig

from fz_openqa import configs
from fz_openqa.datamodules.builders.qa import QaBuilder
from fz_openqa.datamodules.datamodule import DataModule
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer
from fz_openqa.datamodules.analytics import SequenceLengths
from fz_openqa.datamodules.builders.preprocessing.entity import EntityPreprocessing


@hydra.main(
    config_path=str(Path(configs.__file__).parent),
    config_name="script_config.yaml",
)
def run(config: DictConfig) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    datasets.set_caching_enabled(True)

    # initialize the tokenizer
    bert_id = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    tokenizer = init_pretrained_tokenizer(pretrained_model_name_or_path=bert_id)

    # preprocessing
    preprocessing_op = config.get("preprocessing", None)
    if isinstance(preprocessing_op, str):
        preprocessing_op = {"entity": EntityPreprocessing}[preprocessing_op]()

    # initialize the data module
    builder = QaBuilder(
        tokenizer=tokenizer,
        use_subset=config.get("use_subset", False),
        cache_dir=config.sys.cache_dir,
        query_expansion=config.get("query_expansion", None),
        num_proc=config.get("num_proc", 2),
        dset_name=config.get("dset_name", "medqa-us"),
        preprocessing_op=preprocessing_op,
        analytics=[
            SequenceLengths(output_dir="analytics/", verbose=True),
        ],
    )
    dm = DataModule(builder=builder)
    dm.prepare_data()
    dm.setup()
    dm.display_samples(n_samples=3)

    # access dataset
    rich.print(dm.dataset)


if __name__ == "__main__":
    run()
