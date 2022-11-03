from warp_pipes import Batch
from warp_pipes import Eg
from warp_pipes import Pipe


class Batchify(Pipe):
    """Convert an example into a batch"""

    @staticmethod
    def _call_batch(eg: Eg, **kwargs) -> Batch:
        return {k: [v] for k, v in eg.items()}


class DeBatchify(Pipe):
    """Convert a one-element batch into am example"""

    @staticmethod
    def _call_batch(batch: Batch, **kwargs) -> Eg:
        for v in batch.values():
            assert len(v) == 1
        return {k: v[0] for k, v in batch.items()}


class AsBatch(Pipe):
    """Apply a pipe to a single"""

    def __init__(self, pipe: Pipe, **kwargs):
        super(AsBatch, self).__init__(**kwargs)
        self.pipe = pipe

    def _call_batch(self, eg: Eg, **kwargs) -> Eg:
        batch = Batchify._call(eg, **kwargs)
        batch = self.pipe(batch, **kwargs)
        return DeBatchify._call(batch, **kwargs)
