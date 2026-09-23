"""Date-seeded shared-random-number paths and analytic independent-emission peak risk."""
from copy import deepcopy
from typing import TYPE_CHECKING, Literal, NotRequired, TypedDict

import numpy as np
import torch
from torch import Tensor

from gmst.contracts import FloatArray, IntArray, Panel, Prediction
from gmst.features import cal_flags
from gmst.hmm_core import CondHMM, forward_logp, stationary
from gmst.hmm_training import History, io_features, z_features

if TYPE_CHECKING:
    from gmst.baselines import SlotState


class HMMState(TypedDict):
    model: CondHMM
    m: FloatArray
    C: FloatArray
    s_op: FloatArray
    device: str | torch.device
    protocol: str
    kind: str
    K: int
    train_idx: IntArray
    best_epoch: int
    history: list[History]
    inner_state: 'HMMState | None'
    tau: float
    half_life: float
    tuning_status: str
    n_tune: int
    decoder: Literal['constant', 'io']
    kan: bool
    M: int
    lambda_io: float
    lambda_spl: float
    phi_start: float | None
    pc_rate: float
    emission_source: Literal['backbone', 'b1']
    centre_state: NotRequired['SlotState']


def issue_centre(state: HMMState, panel: Panel, day_idx: IntArray) -> FloatArray:
    """Refresh H centres from information available at each requested midnight."""
    from gmst.baselines import predict_slot_median
    m = state['m']
    if 'centre_state' in state:
        m = m.copy()
        days = np.unique(np.concatenate([np.maximum(day_idx-lag, 0) for lag in range(3)]))
        m[days] = predict_slot_median(state['centre_state'], panel, days)
    return m


def day_tensors(model: CondHMM, panel: Panel, days: IntArray, m: FloatArray,
                protocol: str) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    device, dtype = model.W.device, model.W.dtype
    op = cal_flags(panel, protocol)[0]
    z = torch.as_tensor(z_features(panel, protocol, days, kan=model.kan, M=model.M), device=device, dtype=dtype)
    u, v = io_features(panel, m, days, protocol)
    means = torch.as_tensor(m[days], device=device, dtype=dtype)
    operating = torch.as_tensor(np.repeat(op[days, None], 96, axis=1), device=device, dtype=torch.long)
    u_t, v_t = torch.as_tensor(u, device=device, dtype=dtype), torch.as_tensor(v, device=device, dtype=dtype)
    return z, means, operating, u_t, v_t


def filter_last(model: CondHMM, panel: Panel, d: int, m: FloatArray,
                protocol: str = 'A+') -> tuple[Tensor, float]:
    if d < 1:
        raise ValueError('Forecast day must have a preceding calendar day')
    z, means, op, u, v = day_tensors(model, panel, np.array([d-1]), m, protocol)
    y = torch.as_tensor(panel['Y'][d-1:d], device=model.W.device, dtype=model.W.dtype)
    obs = torch.isfinite(y)
    if not obs.any():
        return stationary(model.trans(z)[0, 0]), float('nan')
    return forward_logp(model, y, obs, means, op, z, u, v)[1][0, -1], float(panel['Y'][d-1, -1])


def sigma_re(sigma: FloatArray, phi: FloatArray, s: float) -> FloatArray:
    # ponytail: moment decomposition replaces variance refitting, fit a random-effect likelihood if re changes the conclusion (FR-75)
    remaining = np.maximum(sigma**2 / (1-phi**2) - s**2, 1)
    return np.sqrt(remaining * (1-phi**2))


def day_resid_sd(panel: Panel, train_idx: IntArray, m: FloatArray) -> FloatArray:
    result = np.zeros(2)
    for op in range(2):
        residual = [float(np.nanmean(panel['Y'][d]-m[d])) for d in train_idx if panel['op'][d] == op and np.isfinite(panel['Y'][d]).any()]
        if len(residual) > 1:
            result[op] = np.std(residual, ddof=1)
    return result


@torch.no_grad()
def analytic_risk(model: CondHMM, panel: Panel, d: int, m: FloatArray,
                  C: FloatArray, protocol: str = 'A+',
                  device: str | torch.device | None = None) -> FloatArray:
    """Exact finite-horizon peak CDF for phi=0, evaluated with scaled float64 products."""
    if model.ar:
        raise ValueError('Analytic peak risk requires independent emissions (phi=0)')
    cpu = deepcopy(model).to(device='cpu', dtype=torch.float64)
    pi, _ = filter_last(cpu, panel, d, m, protocol)
    z, mean, op, u, v = day_tensors(cpu, panel, np.array([d]), m, protocol)
    mu, sig, _ = cpu.decode(mean, op, u, v)
    a = cpu.trans(z)[0]
    cdf = 0.5 * (1 + torch.erf((torch.as_tensor(C)[:, None, None] - mu[0]) / (sig[0] * np.sqrt(2))))
    mass = pi.expand(len(C), -1).clone()
    logmass = torch.zeros(len(C), dtype=torch.float64)
    for h in range(96):
        mass = (mass @ a[h]) * cdf[:, h]
        scale = mass.sum(-1)
        logmass += scale.log()
        mass /= scale.clamp_min(torch.finfo(torch.float64).tiny)[:, None]
    return (1-logmass.exp()).clamp(0, 1).numpy()


@torch.no_grad()
def forecast(model: CondHMM, panel: Panel, d: int, m: FloatArray, C: FloatArray,
             protocol: str = 'A+', N: int = 2000, re: bool = False,
             s_op: FloatArray | None = None, seed: int | None = None,
             device: str | torch.device | None = None) -> Prediction:
    if N < 1:
        raise ValueError('Forecast paths must be positive')
    target = model.W.device if device is None else torch.device(device)
    if target.type == 'cuda' and target.index is None:
        target = torch.device('cuda', torch.cuda.current_device())
    if target != model.W.device:
        raise ValueError('Forecast device must match the fitted model')
    pi, y_last = filter_last(model, panel, d, m, protocol)
    z, means, op, u_dec, v_dec = day_tensors(model, panel, np.array([d-1, d]), m, protocol)
    mu, sig, phi = model.decode(means, op, u_dec, v_dec)
    a = model.trans(z[1])
    date_seed = int(panel['dates'][d].strftime('%Y%m%d')) if seed is None else seed
    generator = torch.Generator(device=target).manual_seed(date_seed)
    u0 = torch.rand(N, device=target, dtype=model.W.dtype, generator=generator)
    e0n = torch.randn(N, device=target, dtype=model.W.dtype, generator=generator)
    uniforms = torch.rand((N, 96), device=target, dtype=model.W.dtype, generator=generator)
    eps = torch.randn((N, 96), device=target, dtype=model.W.dtype, generator=generator)
    ure = torch.randn(N, device=target, dtype=model.W.dtype, generator=generator)
    state = torch.searchsorted(pi.cumsum(-1), u0).clamp_max(model.K-1)
    scale = 0.0 if s_op is None else float(s_op[int(op[1, 0])])
    if re:
        variance = (sig.square() / (1-phi.square()).clamp_min(1e-7)-scale**2).clamp_min(1)
        sig = (variance * (1-phi.square())).sqrt()
    residual = y_last-mu[0, -1, state] if np.isfinite(y_last) else e0n*sig[0, -1, state]/(1-phi[0, -1, state].square()).clamp_min(1e-7).sqrt()
    paths = torch.empty((N, 96), device=target, dtype=model.W.dtype)
    shift = ure * scale if re else torch.zeros_like(ure)
    for h in range(96):
        state = (uniforms[:, h, None] > a[h, state].cumsum(-1)).sum(-1).clamp_max(model.K-1)
        residual = phi[1, h, state]*residual + sig[1, h, state]*eps[:, h]
        paths[:, h] = mu[1, h, state] + residual + shift
    samples = paths.cpu().numpy().astype(np.float64)
    maximum = samples.max(1)
    risk = (maximum[:, None] > C).mean(0)
    if not model.ar and not re:
        risk = analytic_risk(model, panel, d, m, C, protocol)
    return {'y_mean': samples.mean(0), 'y_median': np.median(samples, axis=0),
            'q': np.quantile(samples, np.arange(1, 20)/20, axis=0), 'paths': samples,
            'M_hat_median': float(np.median(maximum)), 'M_hat_mean': float(maximum.mean()),
            'peak_time_mode': int(np.bincount(samples.argmax(1), minlength=96).argmax()), 'risk_raw': risk}
