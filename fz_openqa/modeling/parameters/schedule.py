import abc
import math


class Schedule(object):
    def __init__(self, *, num_warmup_steps: int = 0, num_steps: int):
        self.num_warmup_steps = num_warmup_steps
        self.num_steps = num_steps
        self._step = 0

    def step(self):
        self._step += 1

    @abc.abstractmethod
    def __call__(self) -> float:
        ...


class LinearSchedule(Schedule):
    def __init__(self, *, initial_value: float = 0, final_value: float = 1, **kwargs):
        super().__init__(**kwargs)
        self.initial_value = initial_value
        self.final_value = final_value

    def __call__(self) -> float:
        if self._step < self.num_warmup_steps:
            return self.initial_value
        else:
            x = self._step - self.num_warmup_steps
            M = self.num_steps - self.num_warmup_steps
            delta = self.final_value - self.initial_value
            u = min(1.0, x / M)
            return self.initial_value + delta * self.transform(u)

    def transform(self, u: float) -> float:
        return u


class CosineSchedule(LinearSchedule):
    def transform(self, u: float) -> float:
        return 0.5 * (1 - math.cos(math.pi * u))


class StaticSchedule(Schedule):
    def __init__(self, value: float, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    def step(self):
        pass

    def __getitem__(self, item):
        return self.value


def AutoSchedule(*, mode="linear", **kwargs) -> Schedule:
    if mode == "linear":
        return LinearSchedule(**kwargs)
    elif mode == "cosine":
        return CosineSchedule(**kwargs)
    elif mode == "static":
        return StaticSchedule(**kwargs)
    else:
        raise ValueError(f"Unknown schedule mode: {mode}")
