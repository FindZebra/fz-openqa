import logging
import os
from pathlib import Path

import datasets
import hydra
import rich
from omegaconf import DictConfig

from fz_openqa import configs
from fz_openqa.datamodules.analytics.corpus_statistics import ReportCorpusStatistics
from fz_openqa.datamodules.builders.corpus import MedQaCorpusBuilder
from fz_openqa.datamodules.builders.inverse_cloze_task import InverseClozeTaskBuilder
from fz_openqa.datamodules.pipes import TextFormatter
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer
from fz_openqa.utils.pretty import pprint_batch

logger = logging.getLogger(__name__)


@hydra.main(
    config_path=str(Path(configs.__file__).parent),
    config_name="script_config.yaml",
)
def run(config: DictConfig) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    datasets.set_caching_enabled(True)

    # initialize the tokenizer
    tokenizer = init_pretrained_tokenizer(pretrained_model_name_or_path="bert-base-cased")

    # initialize text formatter
    text_formatter = TextFormatter(
        lowercase=True,
        remove_symbols=True,
        remove_ref=True,
        remove_hex=True,
        remove_breaks=True,
    )

    # initialize the corpus builder
    builder = MedQaCorpusBuilder(
        tokenizer=tokenizer,
        to_sentences=config.get("to_sentences", False),
        text_formatter=text_formatter,
        use_subset=config.get("use_subset", False),
        cache_dir=config.sys.cache_dir,
        passage_length=200,
        passage_stride=200,
        num_proc=4,
        analyses=[ReportCorpusStatistics(output_dir="./analyses")],
    )

    # initialize the Inverse Cloze Task Builder
    ict_builder = InverseClozeTaskBuilder(
        corpus_builder=builder, n_neighbours=3, num_proc=4, min_distance=1, poisson_lambda=2
    )
    dataset = ict_builder()
    collate_fn = ict_builder.get_collate_pipe()

    # access dataset
    rich.print(dataset)
    pprint_batch(dataset["train"][:3], "ICT batch")
    batch = collate_fn([dataset["train"][i] for i in range(3)])
    pprint_batch(batch, "ICT batch")

    for i in range(3):
        # todo: replace DOC tokens
        rich.print(ict_builder.format_row({k: v[i] for k, v in batch.items()}))


if __name__ == "__main__":
    run()
