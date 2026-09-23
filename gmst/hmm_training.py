"""Issue-time features, masked training blocks and penalized NLL optimization."""
from copy import deepcopy
from typing import Final, Literal, TypedDict

import numpy as np
import torch
from torch import Tensor

from gmst.contracts import FloatArray, IntArray, Panel
from gmst.features import cal_flags
from gmst.hmm_core import DEVICE, CondHMM, forward_logp, pc_nlog_prior

DZ: Final = {'A': 8, 'A+': 14, 'B': 16}


class Blocks(TypedDict):
    y: Tensor
    obs: Tensor
    m: Tensor
    opt: Tensor
    z: Tensor
    u: Tensor
    v: Tensor
    days: IntArray


class History(TypedDict):
    epoch: int
    train_nll: float
    train_objective: float
    val_nll: float


def cyclic_basis(M: int = 12) -> FloatArray:
    """Periodic cardinal cubic B-splines on the 96-quarter circle."""
    if M < 4:
        raise ValueError('Cubic cyclic splines require at least four knots')
    x = np.arange(96, dtype=float) * M / 96
    centre = np.floor(x).astype(int)
    t = x - centre
    weights = np.stack(((1-t)**3, 3*t**3-6*t*t+4, -3*t**3+3*t*t+3*t+1, t**3), -1) / 6
    basis = np.zeros((96, M))
    for j in range(4):
        basis[np.arange(96), (centre + j - 1) % M] += weights[:, j]
    return basis


def z_features(panel: Panel, protocol: str, day_idx: IntArray,
               *, kan: bool = False, M: int = 12) -> FloatArray:
    op, hol = cal_flags(panel, protocol)
    n = len(day_idx)
    angle = np.arange(96) * (2 * np.pi / 96)
    harmonic = np.stack([f(r * angle) for r in range(1, 4) for f in (np.sin, np.cos)], -1)
    harmonics = np.broadcast_to(harmonic, (n, 96, 6))
    flags = np.broadcast_to(np.stack((panel['dtype'][day_idx] == 1, panel['dtype'][day_idx] == 2, hol[day_idx]), -1)[:, None], (n, 96, 3))
    operating = np.broadcast_to(op[day_idx, None, None], (n, 96, 1))
    if kan:
        b = np.broadcast_to(cyclic_basis(M), (n, 96, M))
        return np.concatenate((b * (1-operating), b * operating, flags), -1).astype(float)
    if protocol == 'A':
        return np.concatenate((harmonics, flags[..., :2]), -1).astype(float)
    design = np.concatenate((harmonics, operating, operating * harmonics[..., :4], flags), -1)
    if protocol == 'B':
        # ponytail: missing production is zero in transitions, add missingness features if suspect-day sensitivity matters (FR-63)
        p = np.nan_to_num(panel['X']['생산량'][day_idx], nan=0)
        design = np.concatenate((design, (p > 0)[..., None], np.log1p(p)[..., None] / 10), -1)
    return design.astype(float)


def io_features(panel: Panel, m: FloatArray, day_idx: IntArray,
                protocol: str = 'A+') -> tuple[FloatArray, FloatArray]:
    """Thirteen decoder inputs and three AR inputs, using strictly previous-day Y."""
    base = z_features(panel, 'A+', day_idx)
    op, hol = cal_flags(panel, protocol)
    base[..., 6] = op[day_idx, None]
    base[..., 7:11] = base[..., :4] * op[day_idx, None, None]
    base[..., 13] = hol[day_idx, None]
    summaries = np.zeros((len(day_idx), 2))
    for i, d in enumerate(day_idx):
        if d > 0:
            residual = panel['Y'][d-1] - m[d-1]
            observed = residual[np.isfinite(residual)]
            if len(observed):
                summaries[i] = observed.mean(), observed.max()
    one = np.ones((*base.shape[:2], 1))
    u = np.concatenate((one, base[..., 6:7], base[..., :4], base[..., 7:9], base[..., 11:14], np.broadcast_to(summaries[:, None], (len(day_idx), 96, 2))), -1)
    return u, np.concatenate((one, base[..., :2]), -1)


def make_blocks(panel: Panel, day_idx: IntArray, m: FloatArray,
                protocol: str, device: str | torch.device = 'cpu',
                dtype: torch.dtype = torch.float32, *, kan: bool = False,
                M: int = 12) -> Blocks | None:
    days = np.array([d for d in day_idx if d >= 1 and np.isfinite(panel['Y'][d]).any()], dtype=np.int64)
    if not len(days):
        return None
    indices = np.stack((days-1, days), -1).ravel()
    op = cal_flags(panel, protocol)[0]
    z = z_features(panel, protocol, indices, kan=kan, M=M)
    u, v = io_features(panel, m, indices, protocol)
    y = torch.as_tensor(panel['Y'][indices].reshape(-1, 192), dtype=dtype, device=device)
    return {'y': y, 'obs': torch.isfinite(y), 'm': torch.as_tensor(m[indices].reshape(-1, 192), dtype=dtype, device=device),
            'opt': torch.as_tensor(np.repeat(op[indices], 96).reshape(-1, 192), dtype=torch.long, device=device),
            'z': torch.as_tensor(z.reshape(len(days), 192, -1), dtype=dtype, device=device),
            'u': torch.as_tensor(u.reshape(len(days), 192, 13), dtype=dtype, device=device),
            'v': torch.as_tensor(v.reshape(len(days), 192, 3), dtype=dtype, device=device), 'days': days}


def block_forward(model: CondHMM, blocks: Blocks) -> tuple[Tensor, Tensor]:
    return forward_logp(model, blocks['y'], blocks['obs'], blocks['m'], blocks['opt'], blocks['z'], blocks['u'], blocks['v'])


def nll(model: CondHMM, blocks: Blocks) -> Tensor:
    return -block_forward(model, blocks)[0][:, 96:].sum() / blocks['obs'][:, 96:].sum()


def occupancy(model: CondHMM, blocks: Blocks) -> Tensor:
    return block_forward(model, blocks)[1][:, 96:].mean((0, 1))


def occ_penalty(occ: Tensor, kappa: float = 0.02, lam: float = 10.0) -> Tensor:
    return lam * torch.relu(kappa-occ).square().sum()


def fit_blocks(model: CondHMM, train: Blocks, val: Blocks | None = None,
               max_epochs: int = 300, patience: int = 20, epochs: int | None = None,
               lr: float = 0.01, occ_floor: bool = False, *, lambda_io: float = 0,
               lambda_spl: float = 0, pc_rate: float = 0) -> tuple[CondHMM, int, list[History]]:
    if max_epochs < 1 or patience < 1 or lr <= 0:
        raise ValueError('Training budget, patience and learning rate must be positive')
    if (epochs is None and val is None) or (epochs is not None and epochs < 1):
        raise ValueError('Early stopping needs tuning blocks; fixed epochs must be positive')
    if pc_rate and (not model.ar or model.io):
        raise ValueError('PC prior requires constant AR coefficients')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_score, best_epoch, best_state = float('inf'), 1, deepcopy(model.state_dict())
    history: list[History] = []
    for epoch in range(1, (epochs if epochs is not None else max_epochs)+1):
        optimizer.zero_grad()
        train_nll = nll(model, train)
        loss = train_nll + model.penalty(lambda_io, lambda_spl)
        if pc_rate:
            loss = loss + pc_nlog_prior(model.phi(), pc_rate) / train['obs'][:, 96:].sum()
        if occ_floor:
            loss = loss + occ_penalty(occupancy(model, train))
        if not torch.isfinite(loss):
            raise FloatingPointError('HMM objective is not finite')
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            score = float(nll(model, val)) if val is not None and epochs is None else float('nan')
        history.append({'epoch': epoch, 'train_nll': float(train_nll.detach()),
                        'train_objective': float(loss.detach()), 'val_nll': score})
        if epochs is None and score < best_score:
            best_score, best_epoch, best_state = score, epoch, deepcopy(model.state_dict())
        if epochs is None and epoch - best_epoch >= patience:
            break
    if epochs is None:
        model.load_state_dict(best_state)
    return model, epochs if epochs is not None else best_epoch, history


def train_hmm(panel: Panel, train_idx: IntArray, val_idx: IntArray, m: FloatArray,
              protocol: str = 'A+', K: int = 3, cond: bool = True, ar: bool = True,
              seed: int = 0, max_epochs: int = 300, patience: int = 20,
              epochs: int | None = None, occ_floor: bool = False,
              device: str | torch.device = DEVICE, *, decoder: Literal['constant', 'io'] = 'constant',
              kan: bool = False, M: int = 12, lambda_io: float = 0.01,
              lambda_spl: float = 0.01, phi_start: float | None = None,
              pc_rate: float = 0) -> tuple[CondHMM, int, list[History]]:
    model = CondHMM(K, 2*M+3 if kan else DZ[protocol], cond, ar, seed, decoder=decoder, kan=kan,
                    M=M, phi_start=phi_start).to(device)
    train = make_blocks(panel, train_idx, m, protocol, device, kan=kan, M=M)
    val = make_blocks(panel, val_idx, m, protocol, device, kan=kan, M=M)
    if train is None:
        raise ValueError('HMM training needs an observed target day with a predecessor')
    return fit_blocks(model, train, val, max_epochs, patience, epochs, occ_floor=occ_floor,
                      lambda_io=lambda_io, lambda_spl=lambda_spl, pc_rate=pc_rate)


def final_epochs(best: list[int]) -> int:
    if not best:
        raise ValueError('Final epochs require pre-September epoch selections')
    return int(np.floor(np.median(best)+0.5))
