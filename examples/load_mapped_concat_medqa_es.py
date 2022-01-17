import os
from pathlib import Path

import datasets
import hydra
import rich
import transformers
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

import fz_openqa
from fz_openqa import configs
from fz_openqa.datamodules.analytics import RetrieverAccuracy
from fz_openqa.datamodules.analytics.count_matched_questions import CountMatchedQuestions
from fz_openqa.datamodules.analytics.plot_match_triggers import PlotTopMatchTriggers
from fz_openqa.datamodules.analytics.plot_retrieval_score_distribution import PlotScoreDistributions
from fz_openqa.datamodules.builders import ConcatQaBuilder
from fz_openqa.datamodules.builders import MedQaCorpusBuilder
from fz_openqa.datamodules.builders import OpenQaBuilder
from fz_openqa.datamodules.datamodule import DataModule
from fz_openqa.datamodules.index.builder import ElasticSearchIndexBuilder
from fz_openqa.datamodules.pipes import ExactMatch
from fz_openqa.datamodules.pipes import TextFormatter
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer
from fz_openqa.utils.config import print_config

OmegaConf.register_new_resolver("whoami", lambda: os.environ.get("USER"))
OmegaConf.register_new_resolver("getcwd", os.getcwd)


@hydra.main(
    config_path=str(Path(configs.__file__).parent),
    config_name="script_config.yaml",
)
def run(config):
    print_config(config)
    # set the context
    datasets.set_caching_enabled(True)
    # datasets.logging.set_verbosity(datasets.logging.CRITICAL)
    transformers.logging.set_verbosity(transformers.logging.CRITICAL)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    seed_everything(1, workers=True)

    # define the default cache location
    default_cache_dir = Path(fz_openqa.__file__).parent.parent / "cache"
    try:
        cache_dir = config["sys"]["cache_dir"]
    except Exception:
        cache_dir = default_cache_dir

    # tokenizer and text formatter
    tokenizer = init_pretrained_tokenizer(pretrained_model_name_or_path="bert-base-cased")
    text_formatter = TextFormatter(lowercase=True)

    # define the medqa builder
    dataset_builder = ConcatQaBuilder(
        tokenizer=tokenizer,
        text_formatter=text_formatter,
        use_subset=config.get("use_subset", True),
        cache_dir=cache_dir,
        num_proc=4,
    )
    dataset_builder.subset_size = [200, 50, 50]

    # define the corpus builder
    corpus_builder = MedQaCorpusBuilder(
        tokenizer=tokenizer,
        text_formatter=text_formatter,
        use_subset=False,
        cache_dir=cache_dir,
        num_proc=4,
    )

    # define the OpenQA builder
    builder = OpenQaBuilder(
        dataset_builder=dataset_builder,
        corpus_builder=corpus_builder,
        index_builder=ElasticSearchIndexBuilder(),
        relevance_classifier=ExactMatch(interpretable=True),
        n_retrieved_documents=1000,
        n_documents=10,
        max_pos_docs=1,
        filter_unmatched=config.get("filter_unmatched", False),
        select_mode=config.get("select_mode", "sample"),
        num_proc=config.get("num_proc", 2),
        batch_size=config.get("batch_size", 10),
        analyses=[
            RetrieverAccuracy(output_dir="./analyses", verbose=True),
        ],
    )

    # define the data module
    dm = DataModule(builder=builder)

    # preprocess the data
    dm.prepare_data()
    dm.setup()
    dm.display_samples(n_samples=3)

    # access dataset
    rich.print(dm.dataset)

    # sample a batch
    # _ = next(iter(dm.train_dataloader()))


if __name__ == "__main__":
    run()
