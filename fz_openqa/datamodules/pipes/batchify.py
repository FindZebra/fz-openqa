from typing import Any
from typing import Dict

from ...utils.datastruct import Batch
from .base import Pipe


class Batchify(Pipe):
    """Convert an example into a batch"""

    @staticmethod
    def __call__(batch: Dict[str, Any], **kwargs) -> Batch:
        return {k: [v] for k, v in batch.items()}


class DeBatchify(Pipe):
    """Convert a one-element batch into am example"""

    @staticmethod
    def __call__(batch: Batch, **kwargs) -> Dict[str, Any]:
        for v in batch.values():
            assert len(v) == 1
        return {k: v[0] for k, v in batch.items()}


class AsBatch(Pipe):
    def __init__(self, pipe: Pipe, **kwargs):
        super(AsBatch, self).__init__(**kwargs)
        self.pipe = pipe

    def __call__(self, eg: Dict[str, Any], **kwargs) -> Batch:
        batch = Batchify.__call__(eg, **kwargs)
        batch = self.pipe(batch, **kwargs)
        return DeBatchify.__call__(batch, **kwargs)
