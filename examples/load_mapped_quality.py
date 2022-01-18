from pathlib import Path

import datasets
import hydra
import rich

from fz_openqa import configs
from fz_openqa.datamodules.analytics import SequenceLengths
from fz_openqa.datamodules.builders import ConcatQaBuilder
from fz_openqa.datamodules.builders import OpenQaBuilder
from fz_openqa.datamodules.builders import QaBuilder
from fz_openqa.datamodules.builders import QuALITYCorpusBuilder
from fz_openqa.datamodules.datamodule import DataModule
from fz_openqa.datamodules.index.builder import StaticIndexBuilder
from fz_openqa.datamodules.pipes import ExactMatch
from fz_openqa.datamodules.pipes import OptionDropout
from fz_openqa.datamodules.pipes import Sampler
from fz_openqa.datamodules.pipes import TextFormatter
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer


@hydra.main(
    config_path=str(Path(configs.__file__).parent),
    config_name="script_config.yaml",
)
def run(config):
    datasets.set_caching_enabled(True)

    # tokenizer and text formatter
    tokenizer = init_pretrained_tokenizer(
        pretrained_model_name_or_path="bert-base-cased", cache_dir=config.sys.cache_dir
    )
    text_formatter = TextFormatter(lowercase=True)

    # define the builder
    Cls = ConcatQaBuilder if config.get("concat", True) else QaBuilder
    dataset_builder = Cls(
        tokenizer=tokenizer,
        text_formatter=text_formatter,
        use_subset=config.get("use_subset", True),
        cache_dir=config.sys.cache_dir,
        dset_name="quality",
        question_length=120,
        num_proc=4,
    )
    dataset_builder.subset_size = [200, 50, 50]

    # define the corpus builder
    corpus_builder = QuALITYCorpusBuilder(
        tokenizer=tokenizer,
        text_formatter=text_formatter,
        use_subset=False,
        cache_dir=config.sys.cache_dir,
        passage_length=200,
        passage_stride=150,
        num_proc=4,
    )

    # define the OpenQA builder
    builder = OpenQaBuilder(
        dataset_builder=dataset_builder,
        corpus_builder=corpus_builder,
        index_builder=StaticIndexBuilder(),
        relevance_classifier=ExactMatch(interpretable=False),
        sampler=Sampler(total=10),
        n_retrieved_documents=200,
        document_nesting_level=1,
        transform=OptionDropout(update=True),
        num_proc=4,
        batch_size=100,
        analytics=[
            SequenceLengths(verbose=True, output_dir="./analytics/"),
        ],
    )

    # define the data module
    dm = DataModule(builder=builder)

    # preprocess the data
    dm.setup()
    dm.display_samples(n_samples=3)

    # access dataset
    rich.print(dm.dataset)


if __name__ == "__main__":
    run()
