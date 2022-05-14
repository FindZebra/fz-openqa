from loguru import logger

from .base import VectorBase
from .faiss import FaissVectorBase
from .torch import TorchVectorBase


def AutoVectorBase(*, index_factory: str, **kwargs) -> VectorBase:
    if index_factory == "torch":
        logger.info("Init TorchVectorBase")
        return TorchVectorBase(index_factory=index_factory, **kwargs)
    else:
        logger.info("Init FaissVectorBase with index_factory: {}".format(index_factory))
        return FaissVectorBase(index_factory=index_factory, **kwargs)
