import numpy as np
import torch
from torch import Tensor

# handle pytorch tensors etc, by using tensorboardX's method

try:
    from tensorboardX.x2num import make_np
except ImportError:

    def make_np(x):
        if isinstance(x, Tensor):
            x = x.cpu().detach().numpy()
        else:
            x = np.array(x)

        return x.copy().astype("float16")


class RunningStats(object):
    """Computes running mean and standard deviation
    Url: https://gist.github.com/wassname/a9502f562d4d3e73729dc5b184db2501
    Adapted from:
        *
        <http://stackoverflow.com/questions/1174984/how-to-efficiently-\
calculate-a-running-standard-deviation>
        * <http://mathcentral.uregina.ca/QQ/database/QQ.09.02/carlos1.html>
        * <https://gist.github.com/fvisin/5a10066258e43cf6acfa0a474fcdb59f>

    Usage:
        rs = RunningStats()
        for i in range(10):
            rs += np.random.randn()
            print(rs)
        print(rs.mean, rs.std)
    """

    def __init__(self, n=0.0, m=None, s=None):
        self.n = n
        self.m = m
        self.s = s

    def clear(self):
        self.n = 0.0

    def push(self, x, per_dim=False):
        x = make_np(x)
        # process input
        if per_dim:
            self.update_params(x)
        else:
            for el in x.flatten():
                self.update_params(el)

    def update_params(self, x):
        self.n += 1
        if self.n == 1:
            self.m = x
            self.s = 0.0
        else:
            prev_m = self.m.copy()
            self.m += (x - self.m) / self.n
            self.s += (x - prev_m) * (x - self.m)

    def __add__(self, other):
        if isinstance(other, RunningStats):
            sum_ns = self.n + other.n
            prod_ns = self.n * other.n
            delta2 = (other.m - self.m) ** 2.0
            return RunningStats(
                sum_ns,
                (self.m * self.n + other.m * other.n) / sum_ns,
                self.s + other.s + delta2 * prod_ns / sum_ns,
            )
        else:
            self.push(other)
            return self

    @property
    def mean(self):
        return self.m if self.n else 0.0

    def variance(self):
        return self.s / (self.n) if self.n else 0.0

    @property
    def std(self):
        return np.sqrt(self.variance())

    def __repr__(self):
        return (
            f"<RunningMean(mean={self.mean}, std={self.std}, n={self.n}, m={self.m}, s={self.s})>"
        )

    def flatten(self):
        return RunningStats(n=self.n, m=self.m.reshape(-1), s=self.s.reshape(-1))

    def cat(self, other):

        # check compatibility
        if not isinstance(other, type(self)):
            raise ValueError(
                f"Incompatible type: {type(other)} != {type(self)}"
                f"self: {self}\n"
                f"other: {other}"
            )
        if other.n != self.n:
            raise ValueError(f"Incompatible n: {other.n} != {self.n}")
        if not isinstance(other.m, type(self.m)):
            raise ValueError(
                f"Incompatible type: {type(other.m)} != {type(self.m)}"
                f"self: {self}\n"
                f"other: {other}"
            )
        if not isinstance(other.s, type(self.s)):
            raise ValueError(
                f"Incompatible type: {type(other.s)} != {type(self.s)}.\n"
                f"self: {self}\n"
                f"other: {other}"
            )

        # concatenate
        cat_op = {
            Tensor: torch.cat,
            np.ndarray: np.concatenate,
        }[type(self.m)]

        return RunningStats(n=self.n, m=cat_op([self.m, other.m]), s=cat_op([self.s, other.s]))
