import os
from pathlib import Path

import datasets
import hydra
import loguru
import rich
from omegaconf import OmegaConf

from fz_openqa import configs
from fz_openqa.datamodules.builders import QaBuilder
from fz_openqa.datamodules.builders.medqa_x_wikipedia_corpus import WikixMedQaCorpusBuilder
from fz_openqa.datamodules.pipes.query_wiki_api import QueryWikiAPI
from fz_openqa.utils.train_utils import setup_safe_env

OmegaConf.register_new_resolver("whoami", lambda: os.environ.get("USER"))
OmegaConf.register_new_resolver("getcwd", os.getcwd)


@hydra.main(
    config_path=str(Path(configs.__file__).parent),
    config_name="script_config.yaml",
)
def run(config):
    datasets.set_caching_enabled(True)
    setup_safe_env()

    # args
    use_subset = config.get("use_subset", False)
    if use_subset:
        subset_size = {
            "train": 100,
            "validation": 100,
            "test": 100,
        }
    else:
        subset_size = None
    num_proc = config.get("num_proc", 4)
    dset_name = config.get("dset_name", "medqa-us+medqa-tw+medmcqa")

    # define the medqa builder
    dataset_builder = QaBuilder(
        tokenizer=None,
        dset_name=dset_name,
        subset_size=subset_size,
        cache_dir=config.sys.cache_dir,
        num_proc=num_proc,
    )

    output_dir = config.get("output_dir", "wikipedia_corpus_v6")
    if use_subset:
        output_dir += "_subset"

    loguru.logger.info(f"Writing to: {output_dir}")

    api = QueryWikiAPI(text_key="answer.text", n_results=config.get("n_results", 10))
    wiki_builder = WikixMedQaCorpusBuilder(
        dataset_builder=dataset_builder,
        query_articles=api,
        directory_name=output_dir,
        cache_dir=config.sys.cache_dir,
        upload_to_drive=False,
        num_proc=num_proc,
        batch_size=10,
    )

    # define the data module
    dataset = wiki_builder()
    rich.print(dataset)


if __name__ == "__main__":
    run()
