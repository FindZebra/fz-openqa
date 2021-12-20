import datasets
from datasets import load_dataset

from .hf_dataset import HfDatasetBuilder


class PubMedQaBuilder(HfDatasetBuilder):
    # HuggingFace dataset id
    dset_script_path_or_id = load_dataset()
