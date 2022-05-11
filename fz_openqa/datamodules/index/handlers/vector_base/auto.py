from loguru import logger

from .base import VectorBase
from .faiss import FaissVectorBase
from .sharded_faiss import ShardedFaissVectorBase
from .torch import TorchVectorBase


def AutoVectorBase(*, index_factory: str, **kwargs) -> VectorBase:
    if index_factory == "torch":
        logger.info("Init TorchVectorBase")
        return TorchVectorBase(index_factory=index_factory, **kwargs)
    elif index_factory.startswith("shard:"):
        index_factory = index_factory.replace("shard:", "")
        logger.info("Init ShardedFaissVectorBase with index_factory: {}".format(index_factory))
        return ShardedFaissVectorBase(index_factory=index_factory, **kwargs)
    else:
        logger.info("Init FaissVectorBase with index_factory: {}".format(index_factory))
        return FaissVectorBase(index_factory=index_factory, **kwargs)
