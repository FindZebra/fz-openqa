import math

import torch
from torch import nn
from torch.distributions import MultivariateNormal
from torch.distributions import Normal


def flat_to_triangular(flat_params, with_diagonal=False):
    L = len(flat_params)
    N = int((-1 + math.sqrt(1 + 8 * L)) // 2)  # matrix size from num params: L = N(N+1)/2
    if not with_diagonal:
        N += 1
    A = torch.zeros((N, N), device=flat_params.device, dtype=flat_params.dtype)
    k = 0
    if with_diagonal:
        for i in range(N):
            A[i, : i + 1] = flat_params[k : k + i + 1]
            k = k + i + 1
    else:
        for i in range(1, N):
            A[i, :i] = flat_params[k : k + i]
            k = k + i
    return A


def triangular_to_flat(A):
    # we don't need this, but if we do we have to implement with_diagonal=False
    N = A.size(0)
    L = (N * (N + 1)) // 2
    flat_params = torch.zeros((L), device=A.device, dtype=A.dtype)
    k = 0
    for i in range(N):
        flat_params[k : k + i + 1] = A[i, : i + 1]
        k = k + i + 1
    return flat_params


def make_cholesky(logvar, cov):
    m = torch.diag(torch.ones_like(logvar))
    std = (logvar / 2).exp()
    return (1 - m) * cov + torch.diag(std)


class MultivariateParameterization(nn.Module):
    """Parametrize `n_parameters` using a Multivariate Gaussian distribution."""

    def __init__(self, n_parameters: int, scale_init: float = 1e-3):
        super(MultivariateParameterization, self).__init__()
        self.n_parameters = n_parameters
        # location parameter
        self.loc = nn.Parameter(torch.zeros((n_parameters)))

        # diagonal covariance
        self.logvar = nn.Parameter((scale_init * torch.ones((n_parameters))).log())

        # off-diagonal covariance
        self.flat_cov = nn.Parameter(
            torch.zeros((n_parameters * (n_parameters - 1) // 2)).normal_() * 0.0
        )

    @property
    def dist(self) -> MultivariateNormal:
        return MultivariateNormal(self.loc, scale_tril=self.choleski)

    def sample(self, *args, **kwargs):
        return self.dist.sample(*args, **kwargs)

    def rsample(self, *args, **kwargs):
        return self.dist.rsample(*args, **kwargs)

    def log_prob(self, *args, **kwargs):
        return self.dist.log_prob(*args, **kwargs)

    def entropy(self, *args, **kwargs):
        return self.dist.entropy(*args, **kwargs)

    @property
    def choleski(self) -> torch.Tensor:
        cov = flat_to_triangular(self.flat_cov)
        L = make_cholesky(self.logvar, cov)
        return L


class DiagonalParameterization(nn.Module):
    """Parametrize `n_parameters` using a Multivariate Gaussian distribution."""

    def __init__(self, n_parameters: int, scale_init: float = 1e-3):
        super(DiagonalParameterization, self).__init__()
        self.n_parameters = n_parameters
        # location parameter
        self.loc = nn.Parameter(torch.zeros((n_parameters)))

        # diagonal covariance
        self.logvar = nn.Parameter((scale_init * torch.ones((n_parameters))).log())

    @property
    def dist(self) -> Normal:
        return Normal(self.loc, self.logvar.exp())


class BayesianLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        gain_init=1.0,
        base_dist: str = "diagonal",
        share_sampled_params: bool = True,
    ):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.share_sampled_params = share_sampled_params

        # total number of parameters
        n_parameters = in_features * out_features
        if self.use_bias:
            n_parameters += out_features

        # compute scale for init (Xavier)
        scale_init = gain_init * math.sqrt(2.0 / float(self.in_features + self.out_features))

        # register parameters
        if base_dist == "multivariate":
            self.BayesianLinear = MultivariateParameterization(n_parameters, scale_init)
        elif base_dist == "diagonal":
            self.BayesianLinear = DiagonalParameterization(n_parameters, scale_init)
        else:
            raise ValueError(f"Unknown base distribution {base_dist}")

    def entropy(self) -> torch.Tensor:
        return self.BayesianLinear.dist.entropy().sum()

    def forward(self, x):
        *bs, hdim = x.shape
        if hdim != self.in_features:
            raise ValueError(
                f"Expected input size {self.in_features}, got {hdim} " f"(input: {x.shape})"
            )

        if self.share_sampled_params:
            bs = [bs[0]]

        # sample the weights of the linear transformation
        W = self.BayesianLinear.dist.rsample(sample_shape=bs)
        if self.use_bias:
            b = W[..., : self.out_features]
            W = W[..., self.out_features :]
        else:
            b = 0

        # reshape weights
        W = W.view(*bs, self.in_features, self.out_features)

        # compute output
        if self.share_sampled_params:
            y = torch.einsum("b...h, bhk -> b...k", x, W)
            if isinstance(b, torch.Tensor):
                b = b.view(*bs, *(1 for _ in range(x.dim() - len(bs) - 1)), b.shape[-1])
                y = y + b

            return y

        else:
            y = torch.einsum("...h, ...hk -> ...k", x, W)
            return y + b
