import logging
import os
import time
from pathlib import Path

import numpy as np
import rich
import torch
from datasets import Dataset
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from torch import nn
from tqdm import tqdm

from fz_openqa.datamodules.index import Index
from fz_openqa.datamodules.index.colbert import ColbertIndex
from fz_openqa.datamodules.pipes import ApplyToAll
from fz_openqa.datamodules.pipes import Collate
from fz_openqa.datamodules.pipes import Sequential
from fz_openqa.utils.pretty import pprint_batch


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


@torch.no_grad()
def run():
    logging.basicConfig(level=logging.DEBUG)
    trainer = Trainer(gpus=4, strategy="dp")
    seed_everything(42)

    # Load the dataset
    N = 10_000  # 520_471_400
    seq_len = 200
    n = N // seq_len
    hdim = 32
    data = {"document.vector": [x for x in torch.randn(n, seq_len, hdim)], "idx": list(range(n))}
    corpus = Dataset.from_dict(data)
    collate_fn = Sequential(Collate(), ApplyToAll(torch.tensor))

    # init index
    index: Index = ColbertIndex(
        dataset=corpus,
        model=IndentityModel(),
        trainer=trainer,
        index_factory="torch",
        faiss_train_size=1_000_000,
        loader_kwargs={
            "batch_size": 1000,
            "num_workers": 2,
            "pin_memory": True,
        },
        required_keys=["vector"],
        model_output_keys=["_hq_", "_hd_"],
        collate_pipe=collate_fn,
        cache_dir=Path(os.getcwd()) / "cache" / "sandbox",
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

    bs = 100
    rich.print(f">> benchmarking: bs={100}")
    for i in tqdm(range(1000), desc=f"Searching bs={bs}", unit="batch"):
        idx = np.random.randint(0, len(corpus), bs)

    rich.print("# END")


def make_query(collate_fn, corpus, idx):
    batch = collate_fn([corpus[int(i)] for i in idx])
    batch["question.vector"] = batch.pop("document.vector")
    return batch


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    run()
