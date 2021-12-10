import logging
import os
from itertools import zip_longest
from pathlib import Path
from typing import Optional

import datasets
import faiss
import hydra
import numpy as np
import pandas as pd
import rich
import torch
import transformers
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from rich.status import Status
from torch import Tensor
from transformers import AutoModel
from transformers import BertTokenizer

import fz_openqa
from fz_openqa import configs
from fz_openqa.callbacks.store_results import StorePredictionsCallback
from fz_openqa.datamodules.builders.corpus import MedQaCorpusBuilder
from fz_openqa.inference.checkpoint import CheckpointLoader
from fz_openqa.modeling.similarities.max_sim import MaxSim
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.pretty import pprint_batch

logger = logging.getLogger(__name__)

DEFAULT_CKPT = "https://drive.google.com/file/d/17XDASu1JYGndCFWNW3zCZGKD2woJuDIg/view?usp=sharing"
# define the default cache location
default_cache_dir = Path(fz_openqa.__file__).parent.parent / "cache"
bert_id = "google/bert_uncased_L-2_H-128_A-2"

OmegaConf.register_new_resolver("whoami", lambda: os.environ.get("USER"))
OmegaConf.register_new_resolver("getcwd", os.getcwd)

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")


class ModelWrapper:
    def __init__(self, trainer: Trainer):
        self.trainer = trainer

    def __call__(self, batch: Batch, **kwargs) -> Batch:
        args = {k: type(v) for k, v in kwargs.items()}
        logger.info(f">> wrapper: batch={type(batch)}, {', '.join(args)}")
        return self.trainer.accelerator.predict_step(batch)


@torch.no_grad()
@hydra.main(
    config_path=str(Path(configs.__file__).parent),
    config_name="script_config.yaml",
)
def run(config: DictConfig) -> None:
    """
    Load a corpus and index it using Faiss. Adapted for ColBERT

    PytorchLightning's Trainer can be used to accelerate indexing.
    Example, to index the whole corpus (~5min on `rhea`):
    ```bash
    poetry run python examples/index_corpus_using_dense_retriever.py \
    trainer.strategy=dp trainer.gpus=8 +batch_size=2000 +num_workers=16
    +n_samples=null +use_subset=False +num_proc=4

    ```
    """
    # set the context
    datasets.set_caching_enabled(True)
    datasets.logging.set_verbosity(datasets.logging.CRITICAL)
    transformers.logging.set_verbosity(transformers.logging.CRITICAL)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    seed_everything(1, workers=True)

    # load model
    loader = CheckpointLoader(config.get("checkpoint", DEFAULT_CKPT), override=config)
    if config.get("verbose", False):
        loader.print_config()

    # Init Lightning trainer
    logger.info(f"Instantiating trainer <{config.trainer.get('_target_', None)}>")
    trainer: Optional[Trainer] = instantiate(
        config.trainer, callbacks=[StorePredictionsCallback(accepted_fields=["document.row_idx"])]
    )
    if isinstance(trainer, (dict, DictConfig)):
        logger.info("No Trainer was provided. PyTorch Lightning acceleration cannot be used.")
        trainer = None

    # set up the corpus builder
    logger.info(f"Initialize corpus <{MedQaCorpusBuilder.__name__}>")
    corpus_builder = MedQaCorpusBuilder(
        tokenizer=loader.tokenizer,
        to_sentences=config.get("to_sentences", False),
        use_subset=config.get("use_subset", True),
        cache_dir=config.get("sys.cache_dir", default_cache_dir),
        num_proc=config.get("num_proc", 2),
    )

    # build the corpus and take a subset
    corpus = corpus_builder()
    n_samples = config.get("n_samples", 1000)
    if n_samples is not None and n_samples > 0:
        n_samples = min(n_samples, len(corpus))
        corpus = corpus.select(range(n_samples))
    batch = corpus[:]
    rich.print(type(batch["document.input_ids"]))
    rich.print(batch["document.row_idx"][5], batch["document.text"][5])

    # load the bert model model
    model = AutoModel.from_pretrained(bert_id)
    model.eval()
    document_representations = model(
        batch["document.input_ids"], batch["document.attention_mask"]
    ).last_hidden_state

    # convert to contiguous numpy array
    document_representations = document_representations[:, 1:-1, :]
    document_representations = document_representations.numpy()
    document_representations = np.ascontiguousarray(document_representations)
    document_representations = document_representations.reshape(-1, 128)

    rich.print(f"Document representations: {document_representations.shape}")

    # faiss parameters + init
    ndims = document_representations.shape[-1]
    nlist = 32  # The number of cells (space partition). Typical value is sqrt(N)
    m = 16  # The number of sub-vector. Typically this is 8, 16, 32, etc.
    n_bits = (
        8  # bits per sub-vector. This is typically 8, so that each sub-vec is encoded by 1 byte
    )
    quantizer = faiss.IndexFlatL2(ndims)
    index = faiss.IndexIVFPQ(
        quantizer, ndims, nlist, m, n_bits, faiss.METRIC_L2
    )  # (quantiser, ndims, nlist, m, faiss.METRIC_L2)
    rich.print(f"Is index trained: {index.is_trained}")

    # Todo: Find a to train the index based on token-level (training is failing)
    with Status("Setting up Faiss Index..."):
        index.train(document_representations)
        index.add(document_representations)

    rich.print(f"Is index trained: {index.is_trained}")

    # add faiss index for each token and store token index to original document
    tok2doc = []
    for idx in batch["document.row_idx"]:
        ids = np.linspace(idx, idx, num=198, dtype="int32").tolist()
        tok2doc.extend(ids)

    rich.print(f"[red]Total number of indices: {index.ntotal}")
    rich.print(f"[red]Number of unique values in tok2doc list: {len(set(tok2doc))}")

    k = 3  # number of retrieved documents
    # query = ["What is the symptons of post polio syndrom?"]
    query = ["Thus, ATP must be continuously synthesized"]
    query_tok = loader.tokenizer(query)
    rich.print(f"[green]Tokenized sequence: {tokenizer.tokenize(query[0])}")
    rich.print(torch.tensor(query_tok["input_ids"]).shape)

    xq = model(
        torch.tensor(query_tok["input_ids"]).detach(),
        torch.tensor(query_tok["attention_mask"]).detach(),
    ).last_hidden_state[:, 1:-1, :]
    # todo: remove padding tokens
    xq = np.ascontiguousarray(xq.numpy())
    xq = xq.reshape(-1, 128)
    rich.print(f"[green] {xq.shape}")

    # Perform search on index
    scores, indices = index.search(xq, k)
    # rich.print(indices)

    indices_flat = [i for sublist in indices for i in sublist]
    scores_flat = [i for sublist in scores for i in sublist]
    doc_indices = [tok2doc[index] for index in indices_flat]

    rich.print(doc_indices, scores_flat)
    for i, idx in enumerate(doc_indices):
        rich.print(f"[red]Document {idx} with score {scores_flat[i]}:[/red] ")
        rich.print(f"[green]{batch['document.text'][idx]}[/green]")
        # rich.print(f"[magenta]{batch['document.input_ids'][idx]}[/magenta]")

    retrieved_input_ids = [batch["document.input_ids"][idx] for idx in doc_indices]
    rich.print(retrieved_input_ids.size)
    retrieved_att_mask = [batch["document.attention_mask"][idx] for idx in doc_indices]
    retrieved_docs = torch.cat(retrieved_input_ids, retrieved_att_mask)

    rich.print(f"shape: {retrieved_docs.shape}, type: {type(retrieved_docs)}")

    # todo: extract document_representations for retrieved documents
    retrieved_doc_reps = model(retrieved_input_ids, retrieved_att_mask).last_hidden_state

    # convert to contiguous numpy array
    retrieved_doc_reps = retrieved_doc_reps[:, 1:-1, :]
    retrieved_doc_reps = retrieved_doc_reps.numpy()
    retrieved_doc_reps = np.ascontiguousarray(retrieved_doc_reps)
    retrieved_doc_reps = retrieved_doc_reps.reshape(-1, 128)

    rich.print(f"Document representations: {retrieved_doc_reps.shape}")
    # apply MaxSim to filter related documents further
    # max_sim(similarity_metric="l2", query=xq, document=document_representations)


if __name__ == "__main__":
    run()