import logging
import os
import sys
from pathlib import Path

sys.path.append(Path(__file__).parent.parent.as_posix())

import datasets
import hydra
import rich
import transformers
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from fz_openqa import configs
from fz_openqa.datamodules.analytics import RetrieverAccuracy
from fz_openqa.datamodules.analytics import RetrieverDistribution
from fz_openqa.datamodules.builders import ConcatQaBuilder
from fz_openqa.datamodules.builders import CorpusBuilder
from fz_openqa.datamodules.builders import OpenQaBuilder
from fz_openqa.datamodules.datamodule import DataModule
from fz_openqa.datamodules.index.builder import ElasticSearchIndexBuilder
from fz_openqa.datamodules.pipes import ExactMatch
from fz_openqa.datamodules.pipes import Sampler
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
    logging.getLogger("elasticsearch").setLevel(logging.WARNING)
    transformers.logging.set_verbosity(transformers.logging.CRITICAL)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    seed_everything(1, workers=True)

    # tokenizer and text formatter
    tokenizer = init_pretrained_tokenizer(pretrained_model_name_or_path="bert-base-cased")
    text_formatter = TextFormatter(lowercase=True)

    # define the medqa builder
    dataset_builder = ConcatQaBuilder(
        tokenizer=tokenizer,
        text_formatter=text_formatter,
        use_subset=config.get("use_subset", True),
        cache_dir=config.sys.cache_dir,
        dset_name=config.get("dset_name", "medqa-us"),
        num_proc=4,
    )
    dataset_builder.subset_size = [200, 50, 50]

    # define the corpus builder
    corpus_builder = CorpusBuilder(
        dset_name=config.get("corpus_name", "medqa"),
        tokenizer=tokenizer,
        text_formatter=text_formatter,
        use_subset=False,
        cache_dir=config.sys.cache_dir,
        num_proc=4,
    )

    # define the OpenQA builder
    builder = OpenQaBuilder(
        dataset_builder=dataset_builder,
        corpus_builder=corpus_builder,
        index_builder=ElasticSearchIndexBuilder(
            auxiliary_weight=config.get("es_aux_weight", 10),
        ),
        relevance_classifier=ExactMatch(interpretable=True),
        n_retrieved_documents=21,
        sampler=Sampler(total=10, largest=True),  # SamplerBoostPositives(total=10, n_boosted=1),
        num_proc=config.get("num_proc", 2),
        batch_size=config.get("batch_size", 100),
        analytics=[
            RetrieverAccuracy(output_dir="./analyses", verbose=True),
            RetrieverDistribution(output_dir="./analyses", verbose=True),
        ],
    )

    # define the data module
    dm = DataModule(builder=builder, num_workers=0)

    # preprocess the data
    dm.setup()

    # access dataset
    rich.print(dm.dataset)

    dm.display_samples(n_samples=10)

    # sample a batch
    # _ = next(iter(dm.train_dataloader()))


if __name__ == "__main__":
    run()
