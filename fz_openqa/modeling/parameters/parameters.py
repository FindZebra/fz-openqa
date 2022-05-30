from __future__ import annotations

from numbers import Number
from typing import Dict

from omegaconf import DictConfig

from fz_openqa.modeling.parameters.schedule import AutoSchedule
from fz_openqa.modeling.parameters.schedule import Schedule
from fz_openqa.modeling.parameters.schedule import StaticSchedule
from fz_openqa.utils import maybe_instantiate


class Parameters:
    """Store a bunch of parameters with their Schedule objects."""

    def __init__(self, **parameters: Dict[None, str, float | Schedule | Dict | DictConfig]):
        self.parameters = {}
        for k, v in parameters.items():
            schedule = self.instantiate_schedule(v)
            self.parameters[k] = schedule

    def instantiate_schedule(self, v):
        if isinstance(v, Schedule):
            schedule = v
        elif isinstance(v, (dict, DictConfig)) and "_target_" in v.keys():
            schedule = maybe_instantiate(v)
        elif isinstance(v, (dict, DictConfig)):
            assert "mode" in v.keys(), "`mode` must be specified to init a `Schedule`."
            schedule = AutoSchedule(**v)
        elif isinstance(v, Number) or v is None:
            schedule = StaticSchedule(v)
        else:
            raise ValueError(f"Unrecognized parameter type: {type(v)}")
        return schedule

    def items(self):
        return self.parameters.items()

    def step(self):
        for k, v in self.parameters.items():
            v.step()

    def __call__(self) -> Dict[str, float]:
        return {k: v() for k, v in self.parameters.items()}

    def __getitem__(self, item):
        return self.parameters[item]()

    def reset(self):
        for k, v in self.parameters.items():
            v.reset()
