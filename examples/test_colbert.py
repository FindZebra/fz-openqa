import logging
import sys
import time
from pathlib import Path

from rich.status import Status

from fz_openqa.datamodules.index.utils.io import log_mem_size

sys.path.append(Path(__file__).parent.parent.as_posix())

import numpy as np
import rich
import torch
from datasets import Dataset
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from torch import nn
from tqdm import tqdm

from fz_openqa import configs
import hydra
from fz_openqa.datamodules.index import Index
from fz_openqa.datamodules.index.colbert import ColbertIndex
from fz_openqa.datamodules.pipes import ApplyToAll
from fz_openqa.datamodules.pipes import Collate
from fz_openqa.datamodules.pipes import Sequential
from fz_openqa.utils.pretty import pprint_batch

# import te omegaconf resolvers
from fz_openqa.training import experiment  # type: ignore


def gen_vectors(corpus_size, seq_len, hdim, dtype, logger=None):
    # Create a dataset
    _bs = 100_000
    _device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    with Status("Initializing vectors"):
        vectors = torch.empty(corpus_size, seq_len, hdim, dtype=dtype)
    log_mem_size(vectors, "vectors", logger=logger)
    # fill values and normalize
    _v_buffer = torch.empty(_bs, seq_len, hdim, dtype=dtype, device=_device)
    for i in tqdm(range(0, len(vectors), _bs), desc="filling with randn and normalizing"):
        _v_buffer.normal_()
        _v_buffer = torch.nn.functional.normalize(_v_buffer, dim=-1)
        v = _v_buffer.clone().to(dtype=vectors.dtype, device=vectors.device)
        vectors[i : i + _bs] = v[: len(vectors[i : i + _bs])]
    return vectors


class IndentityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x, **kwargs):
        output = {}
        if "document.vector" in x.keys():
            output["_hd_"] = x["document.vector"]
        if "question.vector" in x.keys():
            output["_hq_"] = x["question.vector"]
        return output


test_size = 1000
medqa_size = 186_000
medwiki_size = 2_600_000


@torch.no_grad()
@hydra.main(
    config_path=str(Path(configs.__file__).parent),
    config_name="script_config.yaml",
)
def run(config):
    logging.basicConfig(level=logging.DEBUG)
    trainer = Trainer(gpus=4, strategy="dp")
    seed_everything(42)

    # Load the dataset
    n_centroids = 10_000
    N = medwiki_size
    seq_len = 200
    hdim = 64
    vectors = gen_vectors(N, seq_len, hdim, torch.float16, logger=None)
    with Status("Storing vectors as Dataset"):
        data = {"document.vector": [v for v in vectors], "idx": list(range(N))}
        corpus = Dataset.from_dict(data)
    collate_fn = Sequential(Collate(), ApplyToAll(torch.tensor))

    # init index
    index: Index = ColbertIndex(
        dataset=corpus,
        model=IndentityModel(),
        trainer=trainer,
        index_factory=f"shard:IVF{n_centroids},PQ32",
        dtype="float16",
        faiss_train_size=1_000_000,
        loader_kwargs={
            "batch_size": 1000,
            "num_workers": 2,
            "pin_memory": True,
        },
        required_keys=["vector"],
        model_output_keys=["_hq_", "_hd_"],
        collate_pipe=collate_fn,
        cache_dir=Path(config.sys.cache_dir) / "sandbox" / "test-colbert",
        persist_cache=True,
        progress_bar=True,
        p=128,
        nprobe=32,
        max_chunksize=1000,
        maxsim_chunksize=10000,
    )
    index.free_memory()

    # search
    rich.print("[green]=== searching Index ===")
    idx = np.linspace(0, len(corpus) - 1, 1000)
    rich.print(f">> idx={len(idx)}")
    batch = make_query(collate_fn, corpus, idx)
    pprint_batch(batch, f"batch : {type(batch)}")
    # import pdb; pdb.set_trace()
    _ = index(batch, k=10)  # load max_sim
    start_time = time.time()
    output = index(batch, k=1000)
    pprint_batch(output, f"output, elapsed time: {time.time() - start_time:.2f}s")
    rich.print(output["document.row_idx"])
    # del index

    n = 0
    bs = 100
    rich.print(f">> benchmarking: bs={100}")
    hits = []
    ranks = []
    inv_ranks = []
    n_queries = 12_000
    start_time = time.time()
    for i in (pbar := tqdm(range(n_queries // bs), unit="batch")) :
        idx = np.random.randint(0, len(corpus), bs)
        query = make_query(collate_fn, corpus, idx, eps=1e-2)
        output = index(query, k=100)
        out_idx = output["document.row_idx"]
        for k, idx_k in enumerate(idx):
            js = out_idx[k].tolist()
            if idx_k in js:
                r = 1 + js.index(idx_k)
                n += 1
                hits.append(1)
                ranks.append(r)
                inv_ranks.append(1.0 / r)
            else:
                n += 1
                hits.append(0)
                ranks.append(len(js))
                inv_ranks.append(0)

        pbar.set_description(
            f"Searching, "
            f"Hit={np.mean(hits):.2%}, "
            f"Rank={np.mean(ranks):.2f}, "
            f"MRR={np.mean(inv_ranks):.2f}"
        )

    elapsed_time = time.time() - start_time
    rich.print(f"Full search in {elapsed_time:.2f}s, {n_queries / elapsed_time:.2f} Q/s")


def make_query(collate_fn, corpus, idx, eps=0.0):
    batch = collate_fn([corpus[int(i)] for i in idx])
    batch["question.vector"] = batch.pop("document.vector").clone()
    if eps > 0:
        batch["question.vector"] += torch.randn_like(batch["question.vector"]) * eps
    return batch


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  poetry run python examples/test_colbert.py
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    run()
