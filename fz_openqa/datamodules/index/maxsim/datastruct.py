from dataclasses import dataclass
from typing import List
from typing import Optional
from typing import Tuple

from torch import Tensor


@dataclass
class FaissInput:
    q_vectors: Tensor
    p: int
    idx: int = None
    doc_ids: Optional[List[int]] = None


@dataclass
class MaxSimInput:
    q_vectors: Tensor
    pids: Tensor
    k: int
    idx: int = None


@dataclass
class MaxSimStats:
    prop_unique: float
    number_docs_per_row: Tensor


@dataclass
class MaxSimOutput:
    pids: Tensor
    scores: Tensor
    boundaries: Optional[Tuple[int, int]]
    k: int
    idx: int = None
    stats: Optional[MaxSimStats] = None
