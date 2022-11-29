from collections import defaultdict
from typing import Dict
from typing import List
from typing import Optional

import torch
from pydantic import BaseModel
from pydantic import root_validator
from pydantic.fields import Field

STATS_PREFIX = "stats::"
GRAD_PREFIX = "grad::"
METRIC_PREFIX = "metric::"


class TokenizedField(BaseModel):
    """Represents a tokenized text field."""

    class Config:
        arbitrary_types_allowed = True

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: Optional[torch.Tensor] = None

    def stats(self) -> Dict:
        return {"length": float(self.input_ids.shape[-1])}


class DocumentTokenizedField(TokenizedField):
    proposal_score: Optional[torch.Tensor] = Field(
        None,
        description="Retrieval score of document within the proposal distribution",
    )
    proposal_log_weight: Optional[torch.Tensor] = Field(
        None,
        description="Importance-sampling log-weight of the proposal distribution",
    )

    @torch.no_grad()
    def stats(self) -> Dict:
        output = super().stats()
        if self.proposal_score is not None:
            proposal_log_probs = torch.log_softmax(self.proposal_score, dim=-1)

            entropy = -torch.sum(torch.exp(proposal_log_probs) * proposal_log_probs, dim=-1)
            output["proposal_entropy"] = entropy.mean()
        return output


class ReaderRetrieverInputs(BaseModel):
    """Represents the inputs to the ReaderRetriever model."""

    document: Optional[DocumentTokenizedField] = Field(
        None,
        description="Document field (e.g., tokenized Wikipedia passage)",
    )
    question: Optional[TokenizedField] = Field(
        None,
        description="Question field (e.g., tokenized question/query)",
    )
    lm: Optional[TokenizedField] = Field(
        None,
        description="language model input field (e.g., tokenized prompt + expected completion)",
    )

    @root_validator(pre=True)
    def parse_nested_fields(cls, values):
        nested_values = defaultdict(dict)
        for k in list(values.keys()):
            if "." in k:
                v = values.pop(k)
                field, subfield = k.split(".")
                nested_values[field][subfield] = v

        for field, attrs in nested_values.items():
            Cls = ReaderRetrieverInputs.__fields__[field].type_
            values[field] = Cls(**attrs)

        return values

    def flatten(self, keys: Optional[List] = None) -> Dict:
        output = {}
        for k in self.__fields_set__:
            v = getattr(self, k)
            if isinstance(v, TokenizedField):
                v = v.dict()
                keys_ = keys or list(v.keys())
                v = {f"{k}.{ks}": vs for ks, vs in v.items() if ks in keys_}
                output.update(v)
            else:
                output[k] = v

        return output

    def stats(self, fields: Optional[List] = None) -> Dict:
        """Return the stats for each field."""
        output = {}
        if fields is None:
            fields = list(self.__fields_set__)

        for k in list(self.__fields_set__):
            v = getattr(self, k)
            if k not in fields or not isinstance(v, TokenizedField):
                continue

            v = v.stats()
            v = {f"{k}.{ks}": vs for ks, vs in v.items()}
            output.update(v)
        return output
