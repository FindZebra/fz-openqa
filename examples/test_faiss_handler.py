import logging
import os
import sys
import tempfile
from pathlib import Path

from rich.status import Status

sys.path.append(Path(__file__).parent.parent.as_posix())

import hydra
import numpy as np
from tqdm import tqdm

from fz_openqa import configs
from fz_openqa.datamodules.index.handlers.faiss import FaissHandler
from fz_openqa.datamodules.index.utils.io import log_mem_size

# import te omegaconf resolvers
from fz_openqa.training import experiment  # type: ignore

import torch
import faiss
import rich
import time

medqa_size = 186_000
medwiki_size = 2_600_000
ten_million = 10_000_000
one_billion = 1_000_000_000


qa_size = 300 * 20_000  # Q tokens * dataset size
dtype = torch.float16
hdim = 64
seq_len = 200

faiss_train_size = 1_000_000
index_factory = "shard:IVF10000,PQ32"
nprobe = 32
bs = 1_000
k = 128
eps = 1e-2


@hydra.main(
    config_path=str(Path(configs.__file__).parent),
    config_name="script_config.yaml",
)
def run(config):
    cache_dir = config.sys.cache_dir
    with tempfile.TemporaryDirectory(dir=cache_dir) as tmpdir:
        path = Path(tmpdir) / "index.faiss"
        devices = list(range(faiss.get_num_gpus()))

        rich.print(f"tmpdir: {path}, devices={devices}")
        handler = FaissHandler(
            path=path,
            index_factory=index_factory,
            nprobe=nprobe,
            faiss_train_size=faiss_train_size,
        )

        corpus_size = {
            "medqa": medqa_size,
            "medwiki": medwiki_size,
            "ten_million": ten_million,
            "one_billion": one_billion,
        }[config.corpus]

        vectors = gen_vectors(corpus_size)

        # build the index
        handler.build(vectors)
        handler.cpu()
        del handler

        # search the index
        handler = FaissHandler(
            path=path,
            index_factory=index_factory,
            nprobe=nprobe,
            faiss_train_size=faiss_train_size,
        )
        handler.load()
        handler.cuda(devices=devices[len(devices) // 2 :])

        # search the index
        ranks = []
        hit_rate = []
        t0 = time.time()
        for i in (pbar := tqdm(range(0, qa_size, bs), unit="batch")) :
            ids = np.random.randint(0, corpus_size, bs)
            query = vectors[ids].clone()
            query += eps * torch.randn_like(query)
            scores, indices = handler(query, k=k)
            for j, retrieved in zip(ids, indices):
                retrieved = retrieved.cpu().numpy().tolist()
                if j in retrieved:
                    ranks.append(retrieved.index(j))
                    hit_rate.append(1)
                else:
                    ranks.append(len(retrieved))
                    hit_rate.append(0)
            pbar.set_description(
                f"hit={sum(hit_rate) / len(hit_rate):.2%}, "
                f"rank={sum(ranks) / len(ranks):.2f}, "
                f"bs={bs}, k={k}"
            )

        duration = time.time() - t0
        rich.print(
            f"Performed search in {duration:.2f}s, " f"{duration / qa_size * 1e3:.3f} ms/query"
        )

        ids = torch.randint(0, len(vectors), (10,))
        rich.print(f"\n\nExamples: query id: {ids}")
        queries = vectors[ids] + eps * torch.randn_like(vectors[ids])
        scores, indices = handler(queries, k=k)

        ranks = []
        for idx, outs in zip(ids, indices):
            try:
                rank = outs.cpu().numpy().tolist().index(idx.item())
            except ValueError:
                rank = None
            ranks.append(rank)

        rich.print(f"Ranks: {ranks}")


def gen_vectors(corpus_size):
    # Create a dataset
    _bs = 10_000
    with Status("Initializing vectors"):
        vectors = torch.empty(corpus_size * seq_len, hdim, dtype=dtype)
    log_mem_size(vectors, "vectors")
    # fill values and normalize
    for i in tqdm(range(0, len(vectors), _bs), desc="filling with randn and normalizing"):
        v = vectors[i : i + _bs].to("cuda:0")
        v.normal_()
        vectors[i : i + _bs] = torch.nn.functional.normalize(v, dim=-1).cpu()
    return vectors


if __name__ == "__main__":
    run()
