import logging

import en_core_sci_scibert
import rich
from pathlib import Path

import hydra

import fz_openqa
from fz_openqa import configs
from fz_openqa.datamodules.builders.corpus import MedQaCorpusBuilder, logger
from fz_openqa.datamodules.pipes.text_filtering import SciSpacyFilter
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer

from omegaconf import DictConfig

logger = logging.getLogger(__name__)

default_cache_dir = Path(fz_openqa.__file__).parent.parent / "cache"

@hydra.main(
    config_path=str(Path(configs.__file__).parent),
    config_name="script_config.yaml",
)
def run(config: DictConfig) -> None:

    tokenizer = init_pretrained_tokenizer(pretrained_model_name_or_path="bert-base-cased")

    logger.info(f"Initialize corpus <{MedQaCorpusBuilder.__name__}>")
    corpus_builder = MedQaCorpusBuilder(
            tokenizer=tokenizer,
            to_sentences=config.get("to_sentences", False),
            use_subset=config.get("use_subset", True),
            cache_dir=config.get("sys.cache_dir", default_cache_dir),
            num_proc=config.get("num_proc", 2),
        )

    # build the corpus and take a subset
    corpus = corpus_builder()
    n_samples = config.get("n_samples", 10)
    if n_samples is not None and n_samples > 0:
        n_samples = min(n_samples, len(corpus))
        corpus = corpus.select(range(n_samples))
    rich.print(corpus)

    sci = SciSpacyFilter(spacy_model=en_core_sci_scibert, text_key="document.text")

    for i in range(10):
        row = corpus[i]
        rich.print(f"\n[cyan]ScispaCy entities for document #{row['document.row_idx']}:")
        print(sci(batch=[row]))

if __name__ == "__main__":
    run()
