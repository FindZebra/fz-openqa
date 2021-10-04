from typing import Any
from typing import Dict

from ...utils.datastruct import Batch
from .base import Pipe


class Batchify(Pipe):
    """Convert an example into a batch"""

    def __call__(self, batch: Dict[str, Any], **kwargs) -> Batch:
        return {k: [v] for k, v in batch.items()}


class DeBatchify(Pipe):
    """Convert a one-element batch into am example"""

    def __call__(self, batch: Batch, **kwargs) -> Dict[str, Any]:
        for v in batch.values():
            assert len(v) == 1
        return {k: v[0] for k, v in batch.items()}
