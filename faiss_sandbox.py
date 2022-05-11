from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Union, Optional, List

import faiss.contrib.torch_utils  # type: ignore
import rich
from rich.status import Status

from fz_openqa.utils.config import print_config

sys.path.append(Path(__file__).parent.parent.as_posix())

import hydra
import numpy as np
from tqdm import tqdm

from fz_openqa import configs
from fz_openqa.datamodules.index.utils.io import log_mem_size
from loguru import logger

import torch
import faiss
import time

from multiprocessing.dummy import Pool as ThreadPool

Vectors = Union[np.ndarray, torch.Tensor]
FaissMetric = int

test_size = 20_000
medqa_size = 186_000
medwiki_size = 2_600_000
ten_million = 10_000_000
one_billion = 1_000_000_000

qa_size = 12_000
dtype = torch.float16
hdim = 64
seq_len = 200

faiss_train_size = 1_000_000
default_factory = "OPQ16,IVF10000,PQ32"
nprobe = 32
bs = 1_000
k = 128
eps = 1e-2
use_float16 = True
tempmem = 1 << 33


class FaissFactory():
    def __init__(self, factory: str):
        self.factory = factory
        pat = re.compile('(OPQ[0-9]+(_[0-9]+)?|,PCAR[0-9]+)?,' +
                         '(IVF[0-9]+),' +
                         '(PQ[0-9]+|Flat)')

        matchobject = pat.match(factory)

        if not matchobject:
            raise ValueError(f'Could not parse factory string: `{factory}`')

        mog = matchobject.groups()
        self.preproc = mog[0]
        self.ivf = mog[2]
        self.pqflat = mog[3]
        self.n_centroids = int(self.ivf[3:])

    @property
    def clean(self):
        return '-'.join(self.factory.split(','))

    def __repr__(self):
        return (
            f'FaissFactory('
            f'Preproc={self.preproc}, '
            f'IVF={self.ivf}, '
            f'PQ={self.pqflat}, '
            f'centroids={self.n_centroids})'
        )


def sanitize(x: Vectors, force_numpy: bool = False) -> Vectors:
    """ convert array to a c-contiguous float array """
    if isinstance(x, torch.Tensor):
        x = x.to(torch.float32).contiguous()
        if force_numpy:
            x = x.cpu().numpy()
        return x
    elif isinstance(x, np.ndarray):
        return np.ascontiguousarray(x.astype('float32'))
    else:
        raise TypeError(f"{type(x)} is not supported")


def rate_limited_imap(f, l):
    """A threaded imap that does not produce elements faster than they
    are consumed"""
    pool = ThreadPool(1)
    res = None
    for i in l:
        res_next = pool.apply_async(f, (i,))
        if res:
            yield res.get()
        res = res_next
    yield res.get()


def dataset_iterator(x, preproc, bs):
    """ iterate over the lines of x in blocks of size bs"""

    nb = x.shape[0]
    block_ranges = [(i0, min(nb, i0 + bs))
                    for i0 in range(0, nb, bs)]

    def prepare_block(i01):
        i0, i1 = i01
        xb = sanitize(x[i0:i1], force_numpy=True)
        return i0, preproc.apply_py(xb)

    return rate_limited_imap(prepare_block, block_ranges)


def get_gpu_resources(devices=None, tempmem: int = -1):
    gpu_resources = []
    if devices is None:
        ngpu = torch.cuda.device_count()
    else:
        ngpu = len(devices)
    for i in range(ngpu):
        res = faiss.StandardGpuResources()
        if tempmem >= 0:
            res.setTempMemory(tempmem)
        gpu_resources.append(res)

    return gpu_resources


def make_vres_vdev(gpu_resources, i0=0, i1=-1):
    " return vectors of device ids and resources useful for gpu_multiple"
    vres = faiss.GpuResourcesVector()
    vdev = faiss.IntVector()
    if i1 == -1:
        i1 = len(gpu_resources)
    for i in range(i0, i1):
        vdev.push_back(i)
        vres.push_back(gpu_resources[i])
    return vres, vdev


def gen_vectors(corpus_size):
    # Create a dataset
    _bs = 100_000
    _device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    with Status("Initializing vectors"):
        vectors = torch.empty(corpus_size * seq_len, hdim, dtype=dtype)
    log_mem_size(vectors, "vectors", logger=logger)
    # fill values and normalize
    _v_buffer = torch.empty(_bs, hdim, dtype=dtype, device=_device)
    for i in tqdm(range(0, len(vectors), _bs), desc="filling with randn and normalizing"):
        _v_buffer.normal_()
        _v_buffer = torch.nn.functional.normalize(_v_buffer, dim=-1)
        v = _v_buffer.clone().to(dtype=vectors.dtype, device=vectors.device)
        vectors[i: i + _bs] = v
    return vectors


class IdentPreproc:
    """a pre-processor is either a faiss.VectorTransform or an IndentPreproc"""

    def __init__(self, d):
        self.d_in = self.d_out = d

    def apply_py(self, x):
        return x


def train_preprocessor(
        preproc_str,
        *,
        vectors: Vectors,
        n_train: int = 1_000_000
):
    d = vectors.shape[1]
    t0 = time.time()
    if preproc_str.startswith('OPQ'):
        fi = preproc_str[3:-1].split('_')
        m = int(fi[0])
        dout = int(fi[1]) if len(fi) == 2 else d
        preproc = faiss.OPQMatrix(d, m, dout)
    elif preproc_str.startswith('PCAR'):
        dout = int(preproc_str[4:-1])
        preproc = faiss.PCAMatrix(d, dout, 0, True)
    else:
        assert False
    y = sanitize(vectors[:n_train], force_numpy=True)
    logger.info(f"Train Preprocessor: {preproc_str} with vectors of shape {y.shape}")
    preproc.train(y)
    logger.info(f"Preprocessor trained in {(time.time() - t0):.3f}s")
    return preproc


def get_preprocessor(
        faiss_factory: FaissFactory,
        *,
        vectors: Optional[Vectors] = None,
        cache_file: os.PathLike = None,
        n_train: int = 1_000_000
):
    preproc_str = faiss_factory.preproc
    if preproc_str:
        if not cache_file or not os.path.exists(cache_file):
            preproc = train_preprocessor(preproc_str, vectors=vectors, n_train=n_train)
            if cache_file is not None:
                cache_file = Path(cache_file)
                logger.info(f"Storing {type(preproc).__name__} into {cache_file}")
                faiss.write_VectorTransform(preproc, cache_file.as_posix())
        else:
            cache_file = Path(cache_file)
            logger.info(f"Loading Preprocessor from {cache_file}")
            preproc = faiss.read_VectorTransform(cache_file.as_posix())
            logger.info(f"Loaded {type(preproc).__name__}")
    else:
        preproc = IdentPreproc(hdim)
    return preproc


def train_coarse_quantizer(vectors: Vectors,
                           *,
                           n_centroids: int,
                           gpu_resources: List,
                           preproc: faiss.VectorTransform,
                           max_points_per_centroid: int = 10_000_000,
                           n_train: int = 1_000_000,
                           faiss_metric: FaissMetric = faiss.METRIC_INNER_PRODUCT
                           ) -> np.ndarray:
    # get training vectors
    n_train = max(n_train, 256 * n_centroids)
    vectors = sanitize(vectors[:n_train], force_numpy=True)

    # define the Quantizer
    d = preproc.d_out
    clus = faiss.Clustering(d, n_centroids)
    clus.verbose = True
    clus.max_points_per_centroid = max_points_per_centroid

    # preprocess te vectors
    vectors = preproc_vectors(preproc, vectors)

    # move the Quantizer to CUDA
    vres, vdev = make_vres_vdev(gpu_resources=gpu_resources)
    index = faiss.index_cpu_to_gpu_multiple(
        vres, vdev, faiss.IndexFlat(d, faiss_metric))

    # train the Quantizer
    clus.train(vectors, index)
    centroids = faiss.vector_float_to_array(clus.centroids)
    return centroids.reshape(n_centroids, d)


def preproc_vectors(preproc, vectors):
    # preprocess the vectors
    logger.info(f"Apply preproc to vectors {vectors.shape}")
    t0 = time.time()
    vectors = preproc.apply_py(vectors)
    logger.info(f"VectorBase preprocessed in {(time.time() - t0):.3f}s, "
                f"output shape {vectors.shape}")
    return vectors


def prepare_coarse_quantizer(
        faiss_factory: FaissFactory,
        *,
        gpu_resources: List = None,
        preproc: faiss.VectorTransform,
        vectors: Vectors = None,
        cache_file: os.PathLike = None,
        max_points_per_centroid: int = 10_000_000,
        n_train: int = 1_000_000,
        faiss_metric: FaissMetric = faiss.METRIC_INNER_PRODUCT
) -> faiss.IndexFlat:
    n_centroids = faiss_factory.n_centroids
    if cache_file and os.path.exists(cache_file):
        cache_file = Path(cache_file)
        logger.info(f"Loading Centroids from {cache_file}")
        centroids = np.load(cache_file.as_posix())
    else:
        logger.info(f"Training coarse quantizer with {n_centroids} centroids...")
        t0 = time.time()
        centroids = train_coarse_quantizer(
            vectors,
            n_centroids=n_centroids,
            gpu_resources=gpu_resources,
            preproc=preproc,
            max_points_per_centroid=max_points_per_centroid,
            n_train=n_train,
        )
        logger.info(f"Trained coarse quantizer in {time.time() - t0:.3f} s")
        if cache_file is not None:
            cache_file = Path(cache_file)
            logger.info(f"Storing Centroids {centroids.shape} to {cache_file}")
            np.save(cache_file.as_posix(), centroids)

    coarse_quantizer = faiss.IndexFlat(preproc.d_out, faiss_metric)
    coarse_quantizer.add(centroids)

    return coarse_quantizer


def build_and_train_index(vectors: Vectors,
                          *,
                          faiss_factory: FaissFactory,
                          preproc: faiss.VectorTransform,
                          coarse_quantizer: faiss.IndexFlat,
                          n_train: int = 1_000_000,
                          faiss_metric: FaissMetric = faiss.METRIC_INNER_PRODUCT,
                          use_float16: bool = True,
                          ) -> faiss.IndexIVF:
    pqflat_str = faiss_factory.pqflat
    n_centroids = faiss_factory.n_centroids
    d = preproc.d_out
    if pqflat_str == 'Flat':
        logger.info("Making an IVFFlat index")
        ivf_index = faiss.IndexIVFFlat(
            coarse_quantizer, d, n_centroids, faiss_metric
        )
    else:
        m = int(pqflat_str[2:])
        assert m < 56 or use_float16, "PQ%d will work only with -float16" % m
        logger.info("Making an IVFPQ index, m = ", m)
        ivf_index = faiss.IndexIVFPQ(
            coarse_quantizer, d, n_centroids, m, 8)

    coarse_quantizer.this.disown()
    ivf_index.own_fields = True

    # finish training on CPU
    # select vectors
    vectors = sanitize(vectors[:n_train], force_numpy=True)

    # preprocess te vectors
    vectors = preproc_vectors(preproc, vectors)

    t0 = time.time()
    logger.info("Training vector codes (fine quantizer)...")
    ivf_index.train(vectors)
    logger.info(f"Trained fine quantizer in {(time.time() - t0):.3f}s")

    return ivf_index


def populate_index(cpu_index: faiss.IndexIVF,
                   *,
                   preproc: faiss.VectorTransform,
                   vectors: Vectors,
                   gpu_resources: List,
                   max_add_per_gpu=1 << 25,
                   use_float16: bool = True,
                   use_precomputed_tables=False,
                   add_batch_size=65536,
                   ):
    """Add elements to a sharded index. Return the index and if available
    a sharded gpu_index that contains the same data. """

    ngpu = len(gpu_resources)
    if max_add_per_gpu is not None and max_add_per_gpu >= 0:
        max_add = max_add_per_gpu * max(1, ngpu)
    else:
        max_add = len(vectors)

    # cloner options
    co = faiss.GpuMultipleClonerOptions()
    co.useFloat16 = use_float16
    co.useFloat16CoarseQuantizer = False
    co.usePrecomputed = use_precomputed_tables
    co.indicesOptions = faiss.INDICES_CPU
    co.verbose = True
    co.reserveVecs = max_add
    co.shard = True
    assert co.shard_type in (0, 1, 2)

    # define resources and create the GPU shards
    vres, vdev = make_vres_vdev(gpu_resources=gpu_resources)
    gpu_index: faiss.IndexShards = faiss.index_cpu_to_gpu_multiple(vres, vdev, cpu_index, co)

    # add the vectors
    logger.info(f"Populating the idndex with {len(vectors)} vectors")
    t0 = time.time()
    nb = vectors.shape[0]
    for i0, xs in tqdm(dataset_iterator(vectors, preproc, add_batch_size),
                       desc=f"Adding vectors (bs={add_batch_size}, max_add={max_add})",
                       total=nb // add_batch_size):
        i1 = i0 + xs.shape[0]
        gpu_index.add_with_ids(xs, np.arange(i0, i1))
        if max_add > 0 and gpu_index.ntotal > max_add:
            logger.info(f"Reached max_add, flushing to CPU")
            for i in range(ngpu):
                index_src_gpu = faiss.downcast_index(gpu_index.at(i))
                index_src = faiss.index_gpu_to_cpu(index_src_gpu)
                index_src.copy_subset_to(cpu_index, 0, 0, nb)
                index_src_gpu.reset()
                index_src_gpu.reserveMemory(max_add)
            try:
                gpu_index.sync_with_shard_indexes()
            except AttributeError:
                gpu_index.syncWithSubIndexes()

        sys.stdout.flush()
    logger.info("Populating time: %.3f s" % (time.time() - t0))

    logger.info("Aggregate indexes to CPU")
    t0 = time.time()

    if hasattr(gpu_index, 'at'):
        # it is a sharded index
        for i in range(ngpu):
            index_src = faiss.index_gpu_to_cpu(gpu_index.at(i))
            print("  index %d size %d" % (i, index_src.ntotal))
            index_src.copy_subset_to(cpu_index, 0, 0, nb)
    else:
        # simple index
        index_src = faiss.index_gpu_to_cpu(gpu_index)
        index_src.copy_subset_to(cpu_index, 0, 0, nb)

    logger.info("Aggregate indexes done in %.3f s" % (time.time() - t0))

    del gpu_index
    return cpu_index


def get_populated_index(
        cpu_index: faiss.IndexIVF,
        *,
        cache_file: os.PathLike = None,
        **kwargs
):
    if cache_file is None or not os.path.exists(cache_file):
        populated_index = populate_index(cpu_index, **kwargs)
        if cache_file is not None:
            cache_file = Path(cache_file)
            faiss.write_index(populated_index, cache_file.as_posix())

    else:
        cache_file = Path(cache_file)
        populated_index = faiss.read_index(cache_file.as_posix())
    return populated_index


@hydra.main(
    config_path=str(Path(configs.__file__).parent),
    config_name="script_config.yaml",
)
def run(config):
    print_config(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # generate a corpus of vectors
        corpus_size = {
            "test": test_size,
            "medqa": medqa_size,
            "medwiki": medwiki_size,
            "ten_million": ten_million,
            "one_billion": one_billion,
        }[config.get("corpus", "test")]
        vectors = gen_vectors(corpus_size=corpus_size)
        logger.info(f"Loaded vectors of shape: {vectors.shape}")

        # defining the GPU resources
        gpu_resources = get_gpu_resources()
        rich.print(f"GPU resources: {gpu_resources}")

        # test queries
        q_ids = np.random.randint(0, len(vectors), size=10)
        q_vectors = vectors[q_ids].clone()

        # parse the factory string
        faiss_factory = FaissFactory(config.get("factory", default_factory))
        rich.print(faiss_factory)

        ########################################################################
        # train the coarse quantizer (IVF)
        ########################################################################
        preproc_cache_file = tmpdir / f"preproc-{faiss_factory.clean}.faiss"
        preproc = get_preprocessor(faiss_factory,
                                   vectors=vectors,
                                   cache_file=preproc_cache_file
                                   )
        rich.print(preproc)

        # delete and load from the disk
        del preproc
        preproc = get_preprocessor(faiss_factory,
                                   vectors=None,
                                   cache_file=preproc_cache_file)
        rich.print(preproc)
        npq_vectors = sanitize(q_vectors, force_numpy=True)
        rotated = preproc.apply_py(npq_vectors)
        rich.print(f'>> preproc: {q_vectors.shape} -> {rotated.shape} '
                   f'(MSE: {np.mean((rotated - npq_vectors) ** 2)})')

        ########################################################################
        # train the coarse quantizer (IVF)
        ########################################################################
        centroids_cache_file = tmpdir / f"centroids-{faiss_factory.clean}.npy"
        coarse_quantizer = prepare_coarse_quantizer(faiss_factory,
                                                    gpu_resources=gpu_resources,
                                                    preproc=preproc,
                                                    vectors=vectors,
                                                    cache_file=centroids_cache_file,
                                                    )
        rich.print(coarse_quantizer)
        del coarse_quantizer
        coarse_quantizer = prepare_coarse_quantizer(faiss_factory,
                                                    gpu_resources=gpu_resources,
                                                    preproc=preproc,
                                                    vectors=None,
                                                    cache_file=centroids_cache_file,
                                                    )
        rich.print(coarse_quantizer)

        ########################################################################
        # train the fine quantizer (PQ)
        ########################################################################
        cpu_index = build_and_train_index(vectors=vectors,
                                          faiss_factory=faiss_factory,
                                          preproc=preproc,
                                          coarse_quantizer=coarse_quantizer,
                                          )

        index_cache_file = tmpdir / f"index-{faiss_factory.clean}.faiss"
        cpu_index = get_populated_index(cpu_index,
                                        vectors=vectors,
                                        preproc=preproc,
                                        gpu_resources=gpu_resources,
                                        cache_file=index_cache_file,
                                        )
        rich.print(cpu_index)
        rich.print(f">> cpu_index: {cpu_index}, size: {cpu_index.ntotal}")
        del cpu_index
        cpu_index = get_populated_index(None,
                                        cache_file=index_cache_file,
                                        )
        rich.print(f">> cpu_index.loaded: {cpu_index}, size: {cpu_index.ntotal}")

        # exit
        rich.print(f"\n\n>> Cleaning up directory {tmpdir} with content:")
        for f in tmpdir.iterdir():
            rich.print(f" - {f} : size={f.stat().st_size}")


if __name__ == "__main__":
    run()
