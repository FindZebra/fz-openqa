import logging
import time

import faiss
import rich
import torch
from tqdm import tqdm

from fz_openqa.datamodules.index.utils.io import build_emb2pid_from_vectors
from fz_openqa.datamodules.index.utils.io import log_mem_size
from fz_openqa.datamodules.index.utils.maxsim.base_worker import WorkerSignal
from fz_openqa.datamodules.index.utils.maxsim.maxsim import MaxSim


def run():
    corpus_size = 200_000
    vdim = 128
    seq_len = 200
    vectors = torch.randn(size=(corpus_size, seq_len, vdim))
    use_half = False

    log_mem_size(vectors, "vectors")
    emd2pid = build_emb2pid_from_vectors(vectors)
    index = faiss.index_factory(vectors.shape[-1], "IVF100,PQ16x8")
    index = faiss.index_cpu_to_all_gpus(index)
    flat_vecs = vectors.view(-1, vectors.shape[-1])
    rich.print(">> train index..")
    index.train(flat_vecs)
    rich.print(">> add vectors..")
    index.add(flat_vecs)
    index_cpu = faiss.index_gpu_to_cpu(index)
    index.reset()
    del index

    if use_half:
        vectors = vectors.to(torch.float16)

    rich.print(f"> index: {index_cpu}, size={index_cpu.ntotal}")
    # time.sleep(30)

    maxsim = MaxSim(
        token_index=index_cpu,
        vectors=vectors,
        emb2pid=emd2pid,
        max_chunksize=10_000,  # 10_000
        max_queue_size=5,
        ranking_devices=[0, 1, 2, 3, 4, 5],
        faiss_devices=[6, 7],
    )

    maxsim(WorkerSignal.PRINT)
    maxsim.cuda()
    rich.print(">> starting search...")

    n_iter = 1
    q_seq_len = 350
    dset_size = 4 * 12_000
    batch_size = 1000
    n_samples = dset_size // batch_size
    # index = torch.linspace(0, corpus_size - 1, batch_size, dtype=torch.int64)
    # query_vectors = vectors[index]
    query_vectors = torch.randn(size=(batch_size, q_seq_len, vdim))
    rich.print(f"> query vectors: {query_vectors.shape}")
    start_time = time.time()
    for iter_idx in range(n_iter):
        for _ in tqdm(range(n_samples), desc=f"iter {iter_idx}"):
            query_vectors = torch.randn(size=(batch_size, q_seq_len, vdim))
            if use_half:
                query_vectors = query_vectors.to(torch.float16)
            out = maxsim(query_vectors, k=1000, p=100)
            assert out.pids.shape == (
                batch_size,
                1000,
            ), f"output: {out.scores.shape} + {out.pids.shape}"

        break

    rich.print(
        f">> Processed {n_iter * n_samples * batch_size} "
        f"vectors in {time.time() - start_time} seconds. "
        f"{n_iter * n_samples * batch_size / (time.time() - start_time)} "
        f"vectors/sec"
    )
    rich.print("--- end ---")
    maxsim.terminate()
    rich.print("> DONE.")

    rich.print(out.pids[:3])

    rich.print(out.scores[:3])


if __name__ == "__main__":
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    logging.getLogger().setLevel(logging.DEBUG)
    run()
