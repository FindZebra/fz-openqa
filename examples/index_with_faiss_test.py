import logging
import os
from pathlib import Path

import datasets
import faiss
import hydra
import numpy as np
import rich
import torch
from omegaconf import DictConfig
from torch.functional import Tensor
from transformers import AutoModel

import fz_openqa
from fz_openqa import configs
from fz_openqa.datamodules.builders.corpus import MedQaCorpusBuilder
from fz_openqa.datamodules.builders.medqa import MedQABuilder
from fz_openqa.datamodules.datamodule import DataModule
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer

logger = logging.getLogger(__name__)
from fz_openqa.utils.pretty import get_separator
from fz_openqa.utils.pretty import pprint_batch
from fz_openqa.utils.train_utils import setup_safe_env


@hydra.main(
    config_path=str(Path(configs.__file__).parent),
    config_name="script_config.yaml",
)
def run(config: DictConfig) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    datasets.set_caching_enabled(True)

    # define the default cache location
    default_cache_dir = Path(fz_openqa.__file__).parent.parent / "cache"

    # initialize the tokenizer
    tokenizer = init_pretrained_tokenizer(
        pretrained_model_name_or_path="bert-base-cased"
    )

    # initialize the corpus data module
    corpus_builder = MedQaCorpusBuilder(
        tokenizer=tokenizer,
        to_sentences=config.get("to_sentences", False),
        use_subset=config.get("use_subset", True),
        train_batch_size=config.get("train_batch_size", 200),
        cache_dir=config.get("cache_dir", default_cache_dir),
        num_proc=1,
    )
    corpus = DataModule(builder=corpus_builder)
    corpus.prepare_data()
    corpus.setup()
    batch_corpus = next(iter(corpus.train_dataloader()))
    pprint_batch(batch_corpus)

    # initialize the MedQA data module
    dm_builder = MedQABuilder(
        tokenizer=tokenizer,
        use_subset=config.get("use_subset", True),
        train_batch_size=config.get("train_batch_size", 200),
        cache_dir=config.get("cache_dir", default_cache_dir),
        num_proc=1,
    )
    dm = DataModule(builder=dm_builder)
    dm.prepare_data()
    dm.setup()
    batch_dm = next(iter(dm.train_dataloader()))
    pprint_batch(batch_dm)

    # Compute document and question embeddings
    document_encoded_layers, question_encoded_layers = encode_bert(
        document_input_ids=batch_corpus["document.input_ids"].clone().detach(),
        document_attention_mask=batch_corpus["document.attention_mask"]
        .clone()
        .detach(),
        question_input_ids=batch_dm["question.input_ids"].clone().detach(),
        question_attention_mask=batch_dm["question.attention_mask"]
        .clone()
        .detach(),
    )

    d = document_encoded_layers.shape[2]  # dimension
    rich.print(document_encoded_layers[:, 0, :].shape)
    nlist = 1  # number of clusters

    quantiser = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantiser, d, nlist, faiss.METRIC_L2)

    print(f"Index is trained: {index.is_trained}")
    index.train(np.ascontiguousarray(document_encoded_layers[:, 0, :].numpy()))
    print(f"Index is trained: {index.is_trained}")
    index.add(np.ascontiguousarray(document_encoded_layers[:, 0, :].numpy()))
    print(f"Total number of indices: {index.ntotal}")

    k = 3  # number of nearest neighbours
    xq = np.ascontiguousarray(question_encoded_layers[1:2, 0, :].numpy())

    # Perform search on index
    distances, indices = index.search(xq, k)
    print([f'Query: {i}: {batch_dm["question.text"][i]}' for i in indices[0]])
    print(
        [
            f'Passage: {i}: {batch_corpus["document.text"][i]}'
            for i in indices[0]
        ]
    )


def encode_bert(
    document_input_ids: Tensor,
    document_attention_mask: Tensor,
    question_input_ids: Tensor,
    question_attention_mask: Tensor,
):
    """
    Compute document and question embeddings using BERT
    """
    # initialize the bert model
    model = AutoModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
    model.eval()
    with torch.no_grad():
        document_outputs = model(
            input_ids=document_input_ids,
            attention_mask=document_attention_mask,
        )
        document_encoded_layers = document_outputs[0]

        question_outputs = model(
            input_ids=question_input_ids,
            attention_mask=question_attention_mask,
        )
        question_encoded_layers = question_outputs[0]

    return document_encoded_layers, question_encoded_layers


if __name__ == "__main__":
    run()
