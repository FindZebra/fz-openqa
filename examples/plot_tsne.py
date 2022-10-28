import collections
import os
import sys

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

from openTSNE import TSNE
from fz_openqa.training.training import load_checkpoint

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import logging

import datasets
import hydra
import transformers
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from fz_openqa.datamodules.datamodule import DataModule

logger = logging.getLogger(__name__)

import fz_openqa.training.experiment  # type: ignore

from pathlib import Path

import numpy as np
import rich
import torch

from fz_openqa import configs
from fz_openqa.utils.config import print_config
from fz_openqa.utils.tensor_arrow import TensorArrowTable


@hydra.main(
    config_path=str(Path(configs.__file__).parent),
    config_name="eval_retrieval_config.yaml",
)
def run(config):
    print_config(config)
    # set the context
    datasets.set_caching_enabled(True)
    datasets.logging.set_verbosity(datasets.logging.CRITICAL)
    transformers.logging.set_verbosity(transformers.logging.CRITICAL)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_DATASETS_CACHE"] = config.sys.cache_dir
    seed_everything(1, workers=True)

    # load the model
    overrides = DictConfig(OmegaConf.to_container(config.overrides, resolve=True))
    logger.info("overriding config with:")
    print_config(overrides)
    checkpoint_manager = load_checkpoint(
        config.checkpoint,
        override_config=overrides,
        ref_config=config,
        silent=True,
    )

    # instantiate the datamodule
    logger.info(f"Instantiating datamodule <{checkpoint_manager.config.datamodule._target_}>")
    datamodule: DataModule = instantiate(checkpoint_manager.config.datamodule)
    rich.print(datamodule)
    dataset = datamodule.builder.dataset_builder()
    corpus = datamodule.builder.corpus_builder()

    rich.print(dataset)
    rich.print(corpus)

    plot_tsne(dataset["test"], corpus)


@torch.no_grad()
def plot_tsne(
    dataset,
    corpus,
    n_cuis=6,
    pca_components=None,
    perplexity=500,
    n_train=None,
    doc_vectors_path="/scratch/valv/cache/fz-index-50bdffa410fc30a2/vectors/"
    "vectors-74976f49651d00f7-corpus.tsarrow",
    query_vectors_path="/scratch/valv/cache/fz-index-50bdffa410fc30a2/vectors/"
    "vectors-74976f49651d00f7-dset-36c8481ac20d7836.tsarrow/test",
):
    # most common CUIs
    q_cuis = [str(c[0]).lower() for c in dataset["question.cui"]]
    top_cuis = collections.Counter(q_cuis).most_common(n_cuis)
    rich.print(top_cuis)
    top_cuis = [c[0] for c in top_cuis]
    palette = sns.color_palette("hls", n_cuis)
    GRAY = (0.7, 0.7, 0.7)
    DARK_GRAY = (0.3, 0.3, 0.3)
    que_colors = [palette[top_cuis.index(c)] if c in top_cuis else DARK_GRAY for c in q_cuis]
    que_matches = [i for i, c in enumerate(q_cuis) if c in top_cuis]

    # doc CUIs and colors
    doc_cuis = [str(c).lower() for c in corpus["document.cui"]]
    doc_matches = [i for i, c in enumerate(doc_cuis) if c in top_cuis]
    rich.print(f"Matches: {len(doc_matches) / len(doc_cuis):.2%}")
    doc_colors = [palette[top_cuis.index(c)] if c in top_cuis else GRAY for c in doc_cuis]

    doc_vectors = TensorArrowTable(doc_vectors_path)
    que_vectors = TensorArrowTable(query_vectors_path)
    rich.print(doc_vectors)

    X_docs = doc_vectors[:].numpy()
    X_queries = que_vectors[:].numpy()
    X_all = np.concatenate([X_docs, X_queries], axis=0)

    if n_train is not None:
        train_ids = np.random.randint(0, X_all.shape[0], size=n_train)
    else:
        train_ids = slice(None, None)
    X_train = X_all[train_ids]
    if pca_components is not None:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=pca_components)
        pca = pca.fit(X_all)
        X_train = pca.transform(X_train)
    else:
        pca = None

    rich.print("[green]Fitting..")
    model = TSNE(n_components=2, perplexity=perplexity, n_iter=20_000, n_jobs=32, verbose=True)
    model = model.fit(X_train)

    # transform
    rich.print("[green]Transforming..")
    if pca is not None:
        X_docs = pca.transform(X_docs)
        X_queries = pca.transform(X_queries)
    X_docs = model.transform(X_docs)
    X_queries = model.transform(X_queries)

    plt.figure(figsize=(16, 10))

    plt.scatter(X_docs[:, 0], X_docs[:, 1], c=doc_colors[: len(X_docs)], s=1, alpha=0.3)
    plt.scatter(X_queries[:, 0], X_queries[:, 1], c=que_colors[: len(X_queries)], s=50, marker="*")
    plt.scatter(
        X_docs[doc_matches, 0],
        X_docs[doc_matches, 1],
        c=[doc_colors[i] for i in doc_matches],
        s=10,
        alpha=1,
    )
    plt.scatter(
        X_queries[que_matches, 0], X_queries[que_matches, 1], c="black", s=120, alpha=1, marker="*"
    )
    plt.scatter(
        X_queries[que_matches, 0],
        X_queries[que_matches, 1],
        c=[que_colors[i] for i in que_matches],
        s=40,
        alpha=1,
        marker="*",
    )

    plt.grid(False)
    plt.axis("off")
    output_path = Path() / "tsne.png"
    plt.savefig(output_path, dpi=500)
    rich.print(output_path.absolute())


if __name__ == "__main__":
    run()
