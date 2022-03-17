import math
from typing import Optional

import torch
from torch import nn
from torch.distributions import Categorical
from torch.distributions import Distribution
from torch.distributions import kl_divergence
from torch.distributions import MixtureSameFamily
from torch.distributions import MultivariateNormal
from torch.distributions import Normal
from torch.distributions import register_kl


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

    def __init__(
        self,
        layer: nn.Linear,
        log_scale_1: float = -0.0,
        log_scale_2: float = -6.0,
        pi: float = 0.5,
    ):
        super(DiagonalParameterization, self).__init__()
        # location parameter
        loc = layer.weight.view(-1).clone()
        if layer.bias is not None:
            loc = torch.cat([loc, layer.bias.view(-1).clone()])
        self._loc_posterior = nn.Parameter(loc.data)
        self.n_parameters = self._loc_posterior.data.numel()

        # diagonal covariance
        log_scale_init = pi * log_scale_1 + (1 - pi) * log_scale_2
        self._logvar_posterior = nn.Parameter(log_scale_init * torch.ones_like(self._loc_posterior))

        # initial values
        self.register_buffer("_logvar_prior", torch.tensor([log_scale_1, log_scale_2]))
        self.register_buffer("_loc_prior", torch.zeros_like(self._logvar_prior.data))
        self.register_buffer("_prior_mixture_weights", torch.tensor([pi, 1 - pi]))

    @property
    def posterior(self) -> Normal:
        return Normal(self._loc_posterior, self._logvar_posterior.exp())

    @property
    def prior(self) -> Distribution:
        components = Normal(self._loc_prior, self._logvar_prior.exp())
        cat = Categorical(self._prior_mixture_weights)
        return MixtureSameFamily(cat, components)


@register_kl(Normal, MixtureSameFamily)
def kl_normal_mixture_normal(p: Normal, q: MixtureSameFamily):
    p = Normal(p.loc.unsqueeze(-1), p.scale.unsqueeze(-1))
    q = q.expand(p.batch_shape)
    kl = kl_divergence(p, q._component_distribution)
    return (q._mixture_distribution.probs * kl).sum(-1)


class BayesianLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias=True,
        share_sampled_params: bool = False,
    ):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.share_sampled_params = share_sampled_params

        # Initialize the layer from a determinisitc one
        layer = nn.Linear(in_features, out_features, bias=bias)
        self.BayesianLinear = DiagonalParameterization(layer)

    def entropy(self) -> torch.Tensor:
        return self.BayesianLinear.posterior.entropy().sum()

    def kl(self) -> torch.Tensor:
        q = self.BayesianLinear.posterior
        p = self.BayesianLinear.prior
        return kl_divergence(q, p).sum()

    def sample_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Sample weights from the distribution."""
        bs, *dims, hdim = x.shape
        if hdim != self.in_features:
            raise ValueError(
                f"Expected input size {self.in_features}, got {hdim} " f"(input: {x.shape})"
            )
        if self.share_sampled_params:
            return self.BayesianLinear.posterior.rsample()
        else:
            return self.BayesianLinear.posterior.rsample(sample_shape=(bs,))

    def forward(self, x: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:

        if weights is None:
            # when no weights are provided, take the mean of the distribution
            weights = self.BayesianLinear._loc_posterior
        if self.use_bias:
            b = weights[..., : self.out_features]
            W = weights[..., self.out_features :]
        else:
            b = 0
            W = weights

        # reshape weights
        W = W.view(*W.shape[:-1], self.in_features, self.out_features)

        # matrix multiplication
        if self.share_sampled_params:
            y = torch.einsum("...h, hk -> ...k", x, W)
        else:
            y = torch.einsum("b...h, bhk -> b...k", x, W)

        # add the bias if any
        if isinstance(b, torch.Tensor):
            b = b.view(*b.shape[:-1], *(1 for _ in range(x.dim() - b.dim())), b.shape[-1])
            y = y + b

        return y
