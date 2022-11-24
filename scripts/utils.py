import os

import datasets
import transformers
from lightning_lite import seed_everything


def configure_env(config, silent_hf: bool = True):
    datasets.set_caching_enabled(True)
    if silent_hf:
        datasets.logging.set_verbosity(datasets.logging.CRITICAL)
        transformers.logging.set_verbosity(transformers.logging.CRITICAL)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_DATASETS_CACHE"] = str(config.sys.cache_dir)
    os.environ["HF_TRANSFORMERS_CACHE"] = str(config.sys.cache_dir)
    seed_everything(1, workers=True)
