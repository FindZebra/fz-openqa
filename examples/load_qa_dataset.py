import os
import sys
from pathlib import Path

sys.path.append(Path(__file__).parent.parent.as_posix())

import datasets
import hydra
import rich
from omegaconf import DictConfig

from fz_openqa import configs
from fz_openqa.datamodules.builders.qa import QaBuilder, ConcatQaBuilder
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
    if config.get("disable_caching"):
        datasets.set_caching_enabled(False)

    # initialize the tokenizer
    bert_id = config.get("bert_id", "bert-base-uncased")
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
    concat_dset = config.get("concat_qa", False)
    QaBuilderCls = ConcatQaBuilder if concat_dset else QaBuilder
    builder = QaBuilderCls(
        tokenizer=tokenizer,
        subset_size=subset_size,
        cache_dir=config.sys.cache_dir,
        query_expansion=config.get("query_expansion", None),
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
    dm = DataModule(builder=builder)
    dm.prepare_data()
    dm.setup()
    dm.display_samples(n_samples=3, split=config.get("display_split", "train"))

    # access dataset
    rich.print(dm.dataset)


if __name__ == "__main__":
    run()
