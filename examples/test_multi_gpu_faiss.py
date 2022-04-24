import logging
import os
import sys
import tempfile
from pathlib import Path

from tqdm import tqdm

from fz_openqa.datamodules.index.handlers.faiss import FaissHandler

sys.path.append(Path(__file__).parent.parent.as_posix())
import torch
import faiss
import rich

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "index.faiss"
        devices = list(range(faiss.get_num_gpus()))

        rich.print(f"tmpdir: {path}, devices={devices}")
        handler = FaissHandler(
            path=path,
            index_factory="IVF100k,PQ32x8",
            nprobe=32,
            faiss_train_size=None,
            faiss_shard=False,
        )

        # Create a dataset
        vectors = torch.randn(37_306_400, 128)

        # build the index
        handler.build(vectors)
        handler.cpu()
        del handler

        # search the index
        handler = FaissHandler(path=path)
        handler.load()
        handler.cuda(devices=devices[len(devices) // 2 :])

        # search the index
        bs = 10_000
        ranks = []
        precision = []
        for i in (pbar := tqdm(range(0, len(vectors), bs), unit="batch")) :
            ids = list(range(i, i + bs))
            query = vectors[ids]
            scores, indices = handler(query, k=2048)
            for j, retrieved in zip(ids, indices):
                retrieved = retrieved.cpu().numpy().tolist()
                if j in retrieved:
                    ranks.append(retrieved.index(j))
                    precision.append(1)
                else:
                    ranks.append(len(retrieved))
                    precision.append(0)
            pbar.set_description(
                f"prec={sum(precision) / len(precision):.2%}, "
                f"rank={sum(ranks) / len(ranks):.2f}"
            )

        ids = torch.randint(0, len(vectors), (1000,))
        rich.print(f"Searching id: {ids}")
        queries = vectors[ids] + 0 * torch.randn_like(vectors[ids])
        scores, indices = handler(queries, k=2048)

        ranks = []
        for idx, outs in zip(ids, indices):
            try:
                rank = outs.cpu().numpy().tolist().index(idx.item())
            except ValueError:
                rank = None
            ranks.append(rank)

        rich.print(f"Ranks: {ranks}")
