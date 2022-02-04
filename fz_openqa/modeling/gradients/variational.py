from typing import Any
from typing import Dict
from typing import Tuple

import torch
from torch import Tensor

from fz_openqa.modeling.gradients.base import Gradients
from fz_openqa.modeling.gradients.base import Quantities


class VariationalGradients(Gradients):
    """Compute the gradients using a Variational Lower Bound."""

    def compute_loss(
        self, q: Quantities, *, targets: Tensor, **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        lb_logp_a = (q.logp_a__d + q.log_p_d__a.sum(dim=1, keepdim=True)).sum(dim=2)
        lb_logp_a_star = torch.gather(lb_logp_a, dim=1, index=targets[:, None]).squeeze(1)
        return -1 * (lb_logp_a_star).mean(-1), {}
