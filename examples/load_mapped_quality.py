import sys
from pathlib import Path


sys.path.append(Path(__file__).parent.parent.as_posix())

from typing import Optional

from hydra.utils import instantiate
from loguru import logger
from pytorch_lightning import Trainer

from fz_openqa.modeling.zero_shot import ZeroShot

import datasets
import hydra
import rich

from fz_openqa import configs
from fz_openqa.datamodules.analytics import SequenceLengths
from fz_openqa.datamodules.builders import ConcatQaBuilder, CorpusBuilder
from fz_openqa.datamodules.builders import OpenQaBuilder
from fz_openqa.datamodules.datamodule import DataModule
from fz_openqa.datamodules.pipes import TextFormatter
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer
from fz_openqa.datamodules.pipes.sampler import FirstN
from fz_openqa.datamodules.index.builder import ColbertIndexBuilder
from fz_openqa.utils.fingerprint import get_fingerprint


@get_fingerprint.register(ZeroShot)
def _(model: ZeroShot) -> str:
    data = {
        "bert_cfg": model.bert_id,
        "head_cfg": model.head,
        "limit_size": model.limit_size,
    }
    return get_fingerprint(data)


@hydra.main(
    config_path=str(Path(configs.__file__).parent),
    config_name="script_config.yaml",
)
def run(config):
    datasets.set_caching_enabled(True)
    # datasets.logging.set_verbosity(datasets.logging.CRITICAL)

    # Load the tokenizer and the text formatter
    bert_id = "google/bert_uncased_L-2_H-128_A-2"
    tokenizer = init_pretrained_tokenizer(
        pretrained_model_name_or_path=bert_id, cache_dir=config.sys.cache_dir
    )
    text_formatter = TextFormatter(lowercase=True)

    # Load the model
    model = ZeroShot(bert_id=bert_id, head="contextual", limit_size=8, tokenizer=tokenizer)
    model.eval()
    model.freeze()
    logger.info(f"Loaded model {type(model).__name__}, " f"fingerprint={get_fingerprint(model)}")

    # Init the Lightning trainer
    logger.info(f"Instantiating trainer <{config.trainer.get('_target_', None)}>")
    trainer: Optional[Trainer] = instantiate(config.trainer)

    # Define the builder
    dataset_builder = ConcatQaBuilder(
        tokenizer=tokenizer,
        text_formatter=text_formatter,
        use_subset=config.get("use_subset", False),
        cache_dir=config.sys.cache_dir,
        dset_name=config.get("dset_name", "quality"),
        num_proc=config.get("num_proc", 4),
    )
    dataset_builder.subset_size = [200, 50, 50]

    # Define the corpus builder
    corpus_builder = CorpusBuilder(
        tokenizer=tokenizer,
        text_formatter=text_formatter,
        use_subset=False,
        cache_dir=config.sys.cache_dir,
        dset_name=config.get("dset_name", "quality"),
        passage_length=200,
        passage_stride=100,
        num_proc=config.get("num_proc", 4),
    )

    # Define the index builder
    index_builder = ColbertIndexBuilder(
        model=model,
        trainer=trainer,
        model_output_keys=["_hd_", "_hq_"],
        collate_pipe=corpus_builder._get_collate_pipe(),
        loader_kwargs={
            "batch_size": config.get("batch_size", 10),
            "num_workers": config.get("num_workers", 4),
            "pin_memory": config.get("pin_memory", True),
        },
        cache_dir=config.sys.cache_dir,
        persist_cache=True,
        handler="lookup",
    )

    # Define the OpenQA builder
    builder = OpenQaBuilder(
        dataset_builder=dataset_builder,
        corpus_builder=corpus_builder,
        index_builder=index_builder,
        relevance_classifier=None,  # ExactMatch(interpretable=False),
        sampler=FirstN(total=10),
        n_retrieved_documents=300,
        num_proc=config.get("num_proc", 4),
        batch_size=100,
        analytics=[
            SequenceLengths(verbose=True, output_dir="./analytics/"),
        ],
    )

    # Define the data module
    dm = DataModule(
        builder=builder,
        num_workers=config.get("num_workers", 0),
    )

    # Preprocess the data
    dm.setup(keep_in_memory=False, model=model, trainer=trainer)
    dm.display_samples(n_samples=3)

    # Access dataset
    rich.print(dm.dataset)


if __name__ == "__main__":
    run()
