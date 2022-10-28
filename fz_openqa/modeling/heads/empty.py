from __future__ import annotations

from typing import Dict
from typing import Optional

from torch import Tensor

from fz_openqa.modeling.heads.base import Head


class EmptyHead(Head):
    """Pass the proposal score through."""

    def __init__(
        self,
        *,
        reference_key: str = "document.proposal_score",
        **kwargs,
    ):
        super(EmptyHead, self).__init__(**kwargs)
        self.reference_key = reference_key

    @property
    def requires_external_backbone(self) -> bool:
        return False

    def forward(
        self,
        *,
        batch: Dict[str, Tensor] = None,
        **kwargs,
    ) -> Dict:
        """
        return the proposal score as main score.

        Parameters
        ----------
        batch
            Batch of data
        Returns
        -------
        Tensor
            Dict, containing the scores of shape [bs, n_opts, n_docs]
        """
        if self.reference_key not in batch.keys():
            raise KeyError(f"{self.reference_key} not in batch.keys() = {batch.keys()}")

        return {"score": batch[self.reference_key].clone()}

    def preprocess(self, last_hidden_state: Tensor, head: str, **kwargs) -> Tensor:
        return self._preprocess(last_hidden_state, head, **kwargs)

    def _preprocess(
        self, last_hidden_state: Tensor, head: str, mask: Optional[Tensor] = None, **kwargs
    ) -> Tensor:
        return last_hidden_state
