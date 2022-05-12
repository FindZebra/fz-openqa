import sys
import tempfile
import time
from pathlib import Path

import rich
from rich.status import Status

from fz_openqa.utils.tensor_arrow import TensorArrowTable
from fz_openqa.utils.tensor_arrow import TensorArrowWriter

sys.path.append(Path(__file__).parent.parent.as_posix())

import hydra
from tqdm import tqdm

from fz_openqa import configs
from fz_openqa.datamodules.index.utils.io import log_mem_size

# import te omegaconf resolvers
from fz_openqa.training import experiment  # type: ignore

import torch

test_size = 1000
medqa_size = 186_000
medwiki_size = 2_600_000
ten_million = 10_000_000
one_billion = 1_000_000_000

dtype = torch.float16
hdim = 32
seq_len = 200


def gen_vectors(corpus_size, seq_len, hdim, dtype, logger=None):
    # Create a dataset
    _bs = 1_000
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


def yield_vectors(corpus_size, seq_len, hdim, dtype, logger=None):
    # Create a dataset
    _bs = 1_000
    _device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # fill values and normalize
    _v_buffer = torch.empty(_bs, seq_len, hdim, dtype=dtype, device=_device)
    for i in tqdm(range(0, corpus_size, _bs), desc="filling with randn and normalizing"):
        _v_buffer.normal_()
        _v_buffer = torch.nn.functional.normalize(_v_buffer, dim=-1)
        v = _v_buffer.clone().to(dtype=dtype, device=torch.device("cpu"))
        yield v


@hydra.main(
    config_path=str(Path(configs.__file__).parent),
    config_name="script_config.yaml",
)
def run(config):
    cache_dir = Path(config.sys.cache_dir)

    corpus_size = {
        "test": test_size,
        "medqa": medqa_size,
        "medwiki": medwiki_size,
        "ten_million": ten_million,
        "one_billion": one_billion,
    }[config.get("corpus", "test")]

    with tempfile.TemporaryDirectory(dir=cache_dir) as tmp_dir:
        tsr_path = Path(tmp_dir) / "vectors.arrow"
        tsr_path.parent.mkdir(exist_ok=True, parents=True)
        pt_path = Path(tmp_dir) / "torch"
        pt_path.mkdir(exist_ok=True, parents=True)
        with TensorArrowWriter(tsr_path, dtype="float16") as store:
            for i, v in enumerate(yield_vectors(corpus_size, seq_len, hdim, dtype)):
                store.write(v)
                torch.save(v, pt_path / f"{i}.pt")

        # read back
        table = TensorArrowTable(tsr_path, dtype="float16")
        rich.print(f">> Table: {table}")

        rich.print(">> Loading TensorArrowTable")
        t_0 = time.time()
        vectors = table[:]
        t_1 = time.time()
        rich.print(f">> TensorArrowTable: Loaded in {t_1 - t_0} seconds")
        log_mem_size(vectors, "vectors")
        del vectors

        rich.print(">> Loading with torch.load")
        t_0 = time.time()
        vectors = torch.empty(corpus_size, seq_len, hdim, dtype=dtype)
        for i, file in enumerate(pt_path.iterdir()):
            v = torch.load(file.as_posix())
            vectors[i : i + len(v)] = v[: len(vectors[i : i + len(v)])]
        t_1 = time.time()
        rich.print(f">> torch: Loaded in {t_1 - t_0} seconds")


if __name__ == "__main__":
    run()
