"""Differentiable switching AR recursion; missing predecessors use a reset approximation."""
import math
from typing import Final, Literal, assert_never

import torch
from torch import Tensor, nn
from torch.nn import functional as F

DEVICE: Final = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CondHMM(nn.Module):
    """Mutable trainable parameters for the conditional switching Gaussian AR model."""

    def __init__(self, K: int = 3, dz: int = 14, cond: bool = True,
                 ar: bool = True, seed: int = 0, *, decoder: Literal['constant', 'io'] = 'constant',
                 kan: bool = False, M: int = 12, phi_start: float | None = None) -> None:
        super().__init__()
        if K < 1 or dz < 1 or (kan and not cond):
            raise ValueError('K and dz must be positive; splines require conditional transitions')
        if phi_start is not None and (not ar or decoder != 'constant' or not 0 < phi_start < 1):
            raise ValueError('phi_start requires constant positive AR coefficients')
        self.K, self.dz, self.cond, self.ar = K, dz, cond, ar
        self.decoder, self.kan, self.M = decoder, kan, M
        match decoder:
            case 'constant': self.io = False
            case 'io': self.io = True
            case unreachable: assert_never(unreachable)
        g = torch.Generator().manual_seed(seed)
        width = dz if kan else 1 + dz if cond else 1
        w = 0.1 * torch.randn((K, K - 1, width), generator=g)
        if kan:
            w[..., :2 * M] -= 3
        else:
            w[..., 0] -= 3
        self.W = nn.Parameter(w)
        delta = torch.stack((torch.linspace(-2, 2, K), torch.linspace(-20, 20, K)), -1)
        rho = torch.cat((delta[:1], torch.log(torch.expm1(delta[1:] - delta[:-1]))))
        scales = torch.log(torch.expm1(torch.tensor([1.0, 9.0]))).repeat(K, 1)
        if self.io:
            a, b = torch.zeros(K, 13), torch.zeros(K, 13)
            a[:, 0], a[:, 1] = rho[:, 0], rho[:, 1] - rho[:, 0]
            b[:, 0], b[:, 1] = scales[:, 0], scales[:, 1] - scales[:, 0]
            self.a = nn.Parameter(a + 0.001 * torch.randn(a.shape, generator=g))
            self.b = nn.Parameter(b + 0.001 * torch.randn(b.shape, generator=g))
            c = torch.zeros(K, 3)
            c[:, 0] = 1
            self.c = nn.Parameter(c + 0.001 * torch.randn(c.shape, generator=g))
        else:
            self.rho = nn.Parameter(rho + 0.5 * torch.randn(rho.shape, generator=g))
            self.s = nn.Parameter(scales + 0.1 * torch.randn(scales.shape, generator=g))
            if ar:
                initial = 1.0 if phi_start is None else math.log(phi_start / (1 - phi_start))
                self.psi = nn.Parameter(initial + 0.1 * torch.randn(K, generator=g))

    def trans(self, z: Tensor) -> Tensor:
        design = z if self.kan else torch.cat((torch.ones_like(z[..., :1]), z), -1)
        if not self.cond:
            design = design[..., :1]
        off = torch.einsum('...d,ikd->...ik', design, self.W)
        logits = z.new_zeros((*z.shape[:-1], self.K, self.K))
        indices = torch.arange(self.K, device=z.device)
        for i in range(self.K):
            logits[..., i, indices != i] = off[..., i, :]
        return logits.softmax(-1)

    def delta(self) -> Tensor:
        raw = self.rho if not self.io else torch.stack((self.a[:, 0], self.a[:, 0] + self.a[:, 1]), -1)
        return torch.cat((raw[:1], raw[:1] + F.softplus(raw[1:]).cumsum(0)))

    def sigma(self) -> Tensor:
        raw = self.s if not self.io else torch.stack((self.b[:, 0], self.b[:, 0] + self.b[:, 1]), -1)
        return 1 + F.softplus(raw)

    def phi(self) -> Tensor:
        if not self.ar:
            return self.W.new_zeros(self.K)
        return (self.psi if not self.io else self.c[:, 0]).sigmoid()

    def emission(self, m: Tensor, opt: Tensor) -> tuple[Tensor, Tensor]:
        return m[..., None] + self.delta().T[opt], self.sigma().T[opt]

    def decode(self, m: Tensor, opt: Tensor, u: Tensor | None = None,
               v: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        if not self.io:
            mu, sig = self.emission(m, opt)
            return mu, sig, self.phi().expand_as(mu)
        if u is None or v is None:
            raise ValueError('IO decoding requires u and v features')
        raw = u @ self.a.T
        delta = torch.cat((raw[..., :1], raw[..., :1] + F.softplus(raw[..., 1:]).cumsum(-1)), -1)
        phi = (v @ self.c.T).sigmoid() if self.ar else torch.zeros_like(delta)
        return m[..., None] + delta, 1 + F.softplus(u @ self.b.T), phi

    def penalty(self, lambda_io: float = 0, lambda_spl: float = 0) -> Tensor:
        loss = self.W.sum() * 0
        if self.io:
            cols = [2, 3, 4, 5, 6, 7, 11, 12]
            loss = loss + lambda_io * (self.a[:, cols].square().sum() + self.b[:, cols].square().sum())
        if self.kan:
            w = self.W[..., :2 * self.M].reshape(self.K, self.K - 1, 2, self.M)
            loss = loss + lambda_spl * (w - w.roll(1, -1)).square().sum()
        return loss


def pc_rate_for_tail(u: float, alpha: float) -> float:
    """Set the positive AR(1) PC rate from P(phi > u) = alpha."""
    if not (math.isfinite(u) and math.isfinite(alpha) and 0 < u < 1 and 0 < alpha < 1):
        raise ValueError('PC tail requires 0 < u, alpha < 1')
    return -math.log(alpha) / math.sqrt(-math.log1p(-u * u))


def pc_nlog_prior(phi: Tensor, rate: float) -> Tensor:
    """Negative log of the positive AR(1) PC density, including d(phi)/dphi."""
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError('PC prior rate must be positive and finite')
    # ponytail: float64 endpoint clamp, use raw-logit density if fitted sigmoid saturates.
    bounded = phi.to(torch.float64).clamp(min=1e-150, max=1 - torch.finfo(torch.float64).eps)
    log_one_minus = torch.log1p(-bounded.square())
    distance = torch.sqrt(-log_one_minus)
    return (rate * distance - math.log(rate) + log_one_minus + torch.log(distance / bounded)).sum()


def stationary(A: Tensor) -> Tensor:
    """Solve the stationary distribution without a nondifferentiable iteration."""
    k = A.shape[-1]
    lhs = A.transpose(-1, -2) - torch.eye(k, device=A.device, dtype=A.dtype)
    lhs = torch.cat((lhs[..., :-1, :], torch.ones_like(lhs[..., -1:, :])), -2)
    rhs = torch.zeros_like(A[..., 0])
    rhs[..., -1] = 1
    return torch.linalg.solve(lhs, rhs).clamp_min(0)


def pair_factors(model: CondHMM, y: Tensor, obs: Tensor, m: Tensor,
                 opt: Tensor, z: Tensor, u: Tensor | None = None,
                 v: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Pair factors use stationary residual resets immediately after missing Y."""
    # ponytail: stationary AR reset after missing Y, propagate residual mixtures if gap accuracy matters (review R4)
    mu, sig, phi = model.decode(m, opt, u, v)
    y = torch.where(obs, y, torch.zeros_like(y))
    prev_y = torch.cat((y[:, :1], y[:, :-1]), 1)
    prev_mu = torch.cat((mu[:, :1], mu[:, :-1]), 1)
    adjacent = obs & torch.cat((torch.zeros_like(obs[:, :1]), obs[:, :-1]), 1)
    mean_ar = mu.unsqueeze(-2) + phi.unsqueeze(-2) * (prev_y.unsqueeze(-1) - prev_mu).unsqueeze(-1)
    mean = torch.where(adjacent[..., None, None], mean_ar, mu.unsqueeze(-2))
    variance = torch.where(adjacent[..., None], sig.square(), sig.square() / (1 - phi.square()).clamp_min(1e-7))
    logb = -0.5 * ((y[..., None, None] - mean).square() / variance.unsqueeze(-2) + variance.log().unsqueeze(-2) + math.log(2 * math.pi))
    logb = torch.where(obs[..., None, None], logb, torch.zeros_like(logb))
    return model.trans(z).log(), logb


def _forward(loga: Tensor, logb: Tensor) -> tuple[Tensor, Tensor]:
    first = stationary(loga[:, 0].exp()).log() + logb[:, 0, 0]
    scale = first.logsumexp(-1)
    alpha = first - scale[:, None]
    scales, alphas = [scale], [alpha]
    for t in range(1, loga.shape[1]):
        update = (alpha[..., None] + loga[:, t] + logb[:, t]).logsumexp(-2)
        scale = update.logsumexp(-1)
        alpha = update - scale[:, None]
        scales.append(scale)
        alphas.append(alpha)
    return torch.stack(scales, 1), torch.stack(alphas, 1)


def forward_logp(model: CondHMM, y: Tensor, obs: Tensor, m: Tensor,
                 opt: Tensor, z: Tensor, u: Tensor | None = None,
                 v: Tensor | None = None) -> tuple[Tensor, Tensor]:
    logc, logalpha = _forward(*pair_factors(model, y, obs, m, opt, z, u, v))
    return logc, logalpha.exp()


def backward(model: CondHMM, y: Tensor, obs: Tensor, m: Tensor,
             opt: Tensor, z: Tensor, u: Tensor | None = None,
             v: Tensor | None = None) -> Tensor:
    loga, logb = pair_factors(model, y, obs, m, opt, z, u, v)
    logc, logalpha = _forward(loga, logb)
    beta = torch.zeros_like(logalpha[:, -1])
    gammas = [logalpha[:, -1].softmax(-1)]
    for t in range(loga.shape[1] - 1, 0, -1):
        beta = (loga[:, t] + logb[:, t] + beta[:, None, :] - logc[:, t, None, None]).logsumexp(-1)
        gammas.append((logalpha[:, t - 1] + beta).softmax(-1))
    return torch.stack(gammas[::-1], 1)
