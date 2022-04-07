import logging
from pathlib import Path

import datasets
import hydra
import rich

import fz_openqa
import wandb
from fz_openqa import configs
from fz_openqa.datamodules.analytics import LogRetrievedDocuments
from fz_openqa.datamodules.analytics.count_matched_questions import CountMatchedQuestions
from fz_openqa.datamodules.analytics.plot_match_triggers import PlotTopMatchTriggers
from fz_openqa.datamodules.analytics.plot_retrieval_score_distribution import PlotScoreDistributions
from fz_openqa.datamodules.builders import MedQaCorpusBuilder
from fz_openqa.datamodules.builders import OpenQaBuilder
from fz_openqa.datamodules.builders import QaBuilder
from fz_openqa.datamodules.datamodule import DataModule
from fz_openqa.datamodules.index.builder import ElasticSearchIndexBuilder
from fz_openqa.datamodules.pipes import ExactMatch
from fz_openqa.datamodules.pipes import PrioritySampler
from fz_openqa.datamodules.pipes import TextFormatter
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer


@hydra.main(
    config_path=str(Path(configs.__file__).parent),
    config_name="script_config.yaml",
)
def run(config):
    datasets.set_caching_enabled(True)
    logging.getLogger("elasticsearch").setLevel(logging.ERROR)
    datasets.logging.set_verbosity(logging.ERROR)
    wandb.init(project="fz_openqa", group="dev-load_mapped_medqa_es")

    # define the default cache location
    default_cache_dir = Path(fz_openqa.__file__).parent.parent / "cache"

    # tokenizer and text formatter
    tokenizer = init_pretrained_tokenizer(pretrained_model_name_or_path="bert-base-cased")
    text_formatter = TextFormatter(lowercase=True)

    # define the medqa builder
    dataset_builder = QaBuilder(
        tokenizer=tokenizer,
        text_formatter=text_formatter,
        use_subset=config.get("use_subset", True),
        cache_dir=config.get("cache_dir", default_cache_dir),
        num_proc=4,
    )
    dataset_builder.subset_size = [200, 50, 50]

    # define the corpus builder
    corpus_builder = MedQaCorpusBuilder(
        tokenizer=tokenizer,
        text_formatter=text_formatter,
        use_subset=False,
        cache_dir=config.get("cache_dir", default_cache_dir),
        num_proc=4,
    )

    # define the OpenQA builder
    builder = OpenQaBuilder(
        dataset_builder=dataset_builder,
        corpus_builder=corpus_builder,
        index_builder=ElasticSearchIndexBuilder(),
        sampler=PrioritySampler(total=10),
        relevance_classifier=ExactMatch(interpretable=True),
        n_retrieved_documents=100,
        num_proc=2,
        batch_size=50,
        analytics=[
            CountMatchedQuestions(output_dir="./analyses", verbose=True),
            PlotScoreDistributions(output_dir="./analyses", verbose=True),
            PlotTopMatchTriggers(output_dir="./analyses", verbose=True),
            LogRetrievedDocuments(output_dir="./analyses", verbose=True, wandb_log=True),
        ],
    )

    # define the data module
    dm = DataModule(builder=builder)

    # preprocess the data
    dm.prepare_data()
    dm.setup()
    # dm.display_samples(n_samples=3)

    # access dataset
    # rich.print(dm.dataset)

    # sample a batch
    # _ = next(iter(dm.train_dataloader()))


if __name__ == "__main__":
    run()
