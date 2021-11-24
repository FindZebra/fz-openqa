from pathlib import Path

import datasets
import hydra
import rich

import fz_openqa
from fz_openqa import configs
from fz_openqa.datamodules import DataModule
from fz_openqa.datamodules.builders import MedQABuilder
from fz_openqa.datamodules.pipes import TextFormatter
from fz_openqa.datamodules.pipes.query_wiki_api import QueryWikiAPI
from fz_openqa.datamodules.utils.medqa_x_wikipedia_corpus import WikixMedQaCorpusBuilder
from fz_openqa.utils.train_utils import setup_safe_env


@hydra.main(
    config_path=str(Path(configs.__file__).parent),
    config_name="script_config.yaml",
)
def run(config):
    datasets.set_caching_enabled(True)
    setup_safe_env()

    # define the default cache location
    default_cache_dir = Path(fz_openqa.__file__).parent.parent / "cache"

    text_formatter = TextFormatter(
        remove_ref=True,
        remove_linebreaks=True,
    )

    # define the medqa builder
    dataset_builder = MedQABuilder(
        tokenizer=None,
        text_formatter=text_formatter,
        use_subset=config.get("use_subset", False),
        cache_dir=config.get("cache_dir", default_cache_dir),
        num_proc=4,
    )
    dataset_builder.subset_size = [1000, 100, 100]

    file_name = "wikipedia_corpus_v2.arrow"
    if config.get("use_subset", False):
        file_name = file_name.replace(".arrow", "_subset.arrow")
    wiki_builder = WikixMedQaCorpusBuilder(
        dataset_builder=dataset_builder,
        query_articles=QueryWikiAPI(text_key="answer.text"),
        file_name=file_name,
        num_proc=4,
        batch_size=10,
    )

    # define the data module
    dm = DataModule(builder=wiki_builder, train_batch_size=10)

    # prepare both the QA dataset and the corpus
    # dm.prepare_data()
    dm.setup()
    rich.print(dm.dataset)


if __name__ == "__main__":
    run()
