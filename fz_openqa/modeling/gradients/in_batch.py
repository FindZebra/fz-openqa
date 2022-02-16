from typing import Any
from typing import Dict
from typing import Tuple

import torch
from torch import Tensor

from fz_openqa.modeling.gradients.base import Gradients
from fz_openqa.modeling.gradients.base import Quantities


class InBatchGradients(Gradients):
    """Compute the gradients of the option retriever assuming the current
    batch to be the whole dataset."""

    def compute_loss(
        self, q: Quantities, *, targets: Tensor, **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        return -1 * (q.logp_a_star).mean(-1), {}
