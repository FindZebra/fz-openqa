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
from fz_openqa.datamodules.builders.medqa_x_wikipedia_corpus import WikixMedQaCorpusBuilder

# import te omegaconf resolvers
from fz_openqa.training import experiment  # type: ignore


@hydra.main(
    config_path=str(Path(configs.__file__).parent),
    config_name="script_config.yaml",
)
def run(config: DictConfig) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    datasets.set_caching_enabled(True)

    # initialize the data module
    builder = QaBuilder(
        tokenizer=None,
        use_subset=config.get("use_subset", False),
        cache_dir=config.sys.cache_dir,
        query_expansion=None,
        num_proc=config.get("num_proc", 2),
        dset_name=config.get("dset_name", "medqa-us+medqa-tw"),
    )
    qa = builder()

    # access dataset
    rich.print(qa)

    # make the wiki corpus
    medwiki_builder = WikixMedQaCorpusBuilder(
        dataset_builder=builder, cache_dir=config.sys.cache_dir
    )
    medwiki = medwiki_builder()
    rich.print(medwiki)


if __name__ == "__main__":
    run()
