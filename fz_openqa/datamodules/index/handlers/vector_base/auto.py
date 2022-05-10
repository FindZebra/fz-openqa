from loguru import logger

from .base import VectorBase
from .faiss import FaissVectorBase
from .sharded_faiss import ShardedFaissVectorBase
from .torch import TorchVectorBase


def AutoVectorBase(index_factory: str, *args, **kwargs) -> VectorBase:
    if index_factory == "torch":
        logger.info("Building TorchVectorBase")
        return TorchVectorBase(index_factory, *args, **kwargs)
    elif index_factory.startswith("shard:"):
        index_factory = index_factory.replace("shard:", "")
        logger.info("Building ShardedFaissVectorBase with index_factory: {}".format(index_factory))
        return ShardedFaissVectorBase(index_factory, *args, **kwargs)
    else:
        logger.info("Building FaissVectorBase with index_factory: {}".format(index_factory))
        return FaissVectorBase(index_factory, *args, **kwargs)
