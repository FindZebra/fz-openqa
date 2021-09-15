import datasets
import rich
import torch
from datasets import Split

from .base_dm import BaseDataModule
from .datasets import medqa_x_corpus


class MedQAxCorpusDataModule(BaseDataModule):
    """A PyTorch Lightning DataModule wrapping the MedQAxCorpus dataset."""

    dset_script_path_or_id = (
        medqa_x_corpus.__file__  # HuggingFace dataset id or local path to script
    )
    split_ids = [
        datasets.Split.TRAIN,
        datasets.Split.VALIDATION,
        datasets.Split.TEST,
    ]  # split names
    pt_attributes = [
        "idx",
        "passage_idx",
        "input_ids",
        "attention_mask",
        "passage_mask",
    ]  # attributes to be converted into Tensors
    vectors_id = "document.vectors"
