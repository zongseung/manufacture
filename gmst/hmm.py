"""Conditional HMM public API and temporally separated model fitting."""
from datetime import date
from typing import Literal, TypedDict, assert_never

import numpy as np
import polars as pl
import torch

from gmst.backbone import fold_backbone
from gmst.baselines import SlotState
from gmst.contracts import FloatArray, IntArray, Model, Panel, Prediction
from gmst.features import internal_split, role_idx
from gmst.hmm_core import DEVICE, CondHMM, backward, forward_logp, pc_nlog_prior, pc_rate_for_tail, stationary
from gmst.hmm_forecast import (
    HMMState,
    analytic_risk,
    day_resid_sd,
    filter_last,
    forecast,
    issue_centre,
    sigma_re,
)
from gmst.hmm_states import (
    conclusion,
    posthoc_states,
    stability_rows,
    state_stability,
    state_summary,
    transition_table,
)
from gmst.hmm_training import (
    DZ,
    History,
    cyclic_basis,
    fit_blocks,
    io_features,
    make_blocks,
    nll,
    occ_penalty,
    occupancy,
    train_hmm,
    z_features,
)
from gmst.hmm_training import final_epochs as _final_epochs

__all__ = [
    'DEVICE',
    'DZ',
    'CondHMM',
    'HMMFlags',
    'HMMState',
    'analytic_risk',
    'backward',
    'conclusion',
    'cyclic_basis',
    'day_resid_sd',
    'evaluate_nll',
    'filter_last',
    'final_epochs',
    'fit_blocks',
    'forecast',
    'forward_logp',
    'hmm_model',
    'io_features',
    'issue_centre',
    'make_blocks',
    'nll',
    'occ_penalty',
    'occupancy',
    'pc_nlog_prior',
    'pc_rate_for_tail',
    'posthoc_states',
    'select_io_lambda',
    'select_kan',
    'sigma_re',
    'stability_rows',
    'state_stability',
    'state_summary',
    'stationary',
    'train_hmm',
    'transition_table',
    'z_features',
]


def final_epochs(best: list[int]) -> int:
    return _final_epochs(best)


class HMMFlags(TypedDict, total=False):
    N: int
    final_epochs: int
    protocol: str
    K: int
    tau: float
    half_life: int
    max_epochs: int
    patience: int
    seed: int
    device: str | torch.device
    occ_floor: bool
    emission_source: Literal['backbone', 'b1']
    rounds: int
    decoder: Literal['constant', 'io']
    kan: bool
    M: int
    lambda_io: float
    lambda_spl: float
    phi_start: float | None
    pc_rate: float


def _centre(panel: Panel, tr: IntArray, C: FloatArray, flags: HMMFlags) -> tuple[FloatArray, SlotState | None]:
    from gmst.baselines import b1_centre_state
    tau, half_life = flags.get('tau', 10.0), flags.get('half_life', 60)
    if flags.get('emission_source', 'backbone') == 'b1':
        return b1_centre_state(panel, tr, tau, half_life, C, flags.get('protocol', 'A+'), flags.get('rounds', 100))
    return fold_backbone(panel, tr, tau, half_life, flags.get('protocol', 'A+')), None


def _train(panel: Panel, tr: IntArray, tune: IntArray, m: FloatArray,
           flags: HMMFlags, kind: Literal['B2', 'B3', 'B4'], epochs: int | None = None) -> tuple[CondHMM, int, list[History]]:
    match kind:
        case 'B2': cond, ar = False, False
        case 'B3': cond, ar = True, False
        case 'B4': cond, ar = True, True
        case unreachable: assert_never(unreachable)
    return train_hmm(panel, tr, tune, m, flags.get('protocol', 'A+'),
        flags.get('K', 3), cond, ar, flags.get('seed', 0),
        flags.get('max_epochs', 300), flags.get('patience', 20), epochs,
        flags.get('occ_floor', False), flags.get('device', DEVICE),
        decoder=flags.get('decoder', 'constant'), kan=flags.get('kan', False),
        M=flags.get('M', 12), lambda_io=flags.get('lambda_io', .01),
        lambda_spl=flags.get('lambda_spl', .01), phi_start=flags.get('phi_start'),
        pc_rate=flags.get('pc_rate', 0))


def evaluate_nll(state: HMMState, panel: Panel, day_idx: IntArray) -> tuple[float, int]:
    """Read outer observations only when the evaluator explicitly requests a score."""
    model = state['model']
    blocks = make_blocks(panel, day_idx, issue_centre(state, panel, day_idx), state['protocol'], model.W.device, model.W.dtype, kan=model.kan, M=model.M)
    if blocks is None:
        return float('nan'), 0
    with torch.no_grad():
        return float(nll(model, blocks)), int(blocks['obs'][:, 96:].sum())


def hmm_model(kind: Literal['B2', 'B3', 'B4'] = 'B4', protocol: str = 'A+', K: int = 3,
              re: bool = False, tau: float = 10, half_life: int = 60,
              final_epochs: int | None = None, max_epochs: int = 300, patience: int = 20,
              N: int = 2000, train_from: date | None = None, occ_floor: bool = False,
              seed: int = 0, device: str | torch.device = DEVICE, *,
              emission_source: Literal['backbone', 'b1'] = 'backbone', rounds: int = 100,
              decoder: Literal['constant', 'io'] = 'constant', kan: bool = False,
              M: int = 12, lambda_io: float = 0.01, lambda_spl: float = 0.01,
              phi_start: float | None = None, pc_rate: float = 0) -> Model[HMMState]:
    if pc_rate < 0 or not np.isfinite(pc_rate):
        raise ValueError('PC prior rate must be nonnegative and finite')
    if pc_rate and (kind != 'B4' or decoder != 'constant'):
        raise ValueError('PC prior is only supported for constant B4 AR coefficients')
    if phi_start is not None and (kind != 'B4' or decoder != 'constant'):
        raise ValueError('phi_start is only supported for constant B4 AR coefficients')
    flags: HMMFlags = {'protocol': protocol, 'K': K, 'tau': tau, 'half_life': half_life,
        'max_epochs': max_epochs, 'patience': patience, 'seed': seed, 'device': device,
        'occ_floor': occ_floor, 'emission_source': emission_source, 'rounds': rounds,
        'decoder': decoder, 'kan': kan, 'M': M, 'lambda_io': lambda_io, 'lambda_spl': lambda_spl,
        'phi_start': phi_start, 'pc_rate': pc_rate}

    def fit(panel: Panel, fold: str, C: FloatArray) -> HMMState:
        from gmst.baselines import predict_slot_median
        tr = role_idx(panel, fold, 'train')
        if train_from is not None:
            tr = np.array([d for d in tr if panel['dates'][d] >= train_from], dtype=np.int64)
        fit_idx, tune, cal = internal_split(panel, fold, tr)
        best = min(20, max_epochs)
        history: list[History] = []
        status = 'default_empty'
        if fold == 'test':
            if final_epochs is None:
                raise ValueError('Final HMM fit requires frozen pre-September epochs')
            best, status = final_epochs, 'final_fixed'
        elif len(tune):
            m_fit, centre = _centre(panel, fit_idx, C, flags)
            if centre is not None:
                days = np.unique(np.maximum(np.r_[tune, tune-1, tune-2], 0))
                m_fit[days] = predict_slot_median(centre, panel, days)
            _, best, history = _train(panel, fit_idx, tune, m_fit, flags, kind)
            status = 'tuned'

        def refit(indices: IntArray) -> HMMState:
            m, centre = _centre(panel, indices, C, flags)
            model = _train(panel, indices, np.array([], dtype=np.int64), m, flags, kind, best)[0]
            state: HMMState = {'model': model, 'm': m, 'C': C.copy(), 's_op': day_resid_sd(panel, indices, m),
                'device': device, 'protocol': protocol, 'kind': kind, 'K': K, 'train_idx': indices,
                'best_epoch': best, 'history': history, 'inner_state': None, 'tau': tau, 'half_life': half_life,
                'tuning_status': status, 'n_tune': len(tune), 'decoder': decoder, 'kan': kan,
                'M': M, 'lambda_io': lambda_io, 'lambda_spl': lambda_spl,
                'phi_start': phi_start, 'pc_rate': pc_rate, 'emission_source': emission_source}
            if centre is not None:
                state['centre_state'] = centre
            return state

        inner = refit(tr[tr < cal[0]]) if len(cal) else None
        state = refit(tr)
        state['inner_state'] = inner
        return state

    def predict(state: HMMState, panel: Panel, d: int) -> Prediction:
        m = issue_centre(state, panel, np.array([d], dtype=np.int64))
        return forecast(state['model'], panel, d, m, state['C'], state['protocol'], N, re, state['s_op'], device=state['device'])

    return {'name': kind, 'fit': fit, 'predict': predict, 'posthoc': posthoc_states,
            'inner_state': lambda state: state['inner_state'], 'evaluate_nll': evaluate_nll}


def _grid_score(panel: Panel, flags: HMMFlags, fold: str) -> tuple[float, int]:
    from gmst.baselines import predict_slot_median
    tr = role_idx(panel, fold, 'train')
    fit_idx, tune, _ = internal_split(panel, fold, tr)
    if not len(tune):
        return float('nan'), 0
    m, centre = _centre(panel, fit_idx, np.zeros(3), flags)
    if centre is not None:
        days = np.unique(np.maximum(np.r_[tune, tune-1, tune-2], 0))
        m[days] = predict_slot_median(centre, panel, days)
    model, _, _ = _train(panel, fit_idx, tune, m, flags, 'B4')
    blocks = make_blocks(panel, tune, m, flags.get('protocol', 'A+'), model.W.device, model.W.dtype, kan=model.kan, M=model.M)
    if blocks is None:
        return float('nan'), 0
    with torch.no_grad():
        return float(nll(model, blocks)), len(blocks['days'])


def select_io_lambda(panel: Panel, flags: HMMFlags, fold: str,
                     grid: tuple[float, ...] = (0, 0.01, 0.1, 1)) -> tuple[float, pl.DataFrame]:
    rows: list[tuple[str, float, float, int, str]] = []
    if not len(internal_split(panel, fold)[1]):
        return 0.01, pl.DataFrame([(fold, 0.01, float('nan'), 0, 'default_empty')], schema=['fold', 'lambda_io', 'nll_tune', 'n_tune', 'status'], orient='row')
    for lam in grid:
        chosen: HMMFlags = {**flags, 'decoder': 'io', 'lambda_io': lam}
        score, n = _grid_score(panel, chosen, fold)
        rows.append((fold, lam, score, n, 'tuned'))
    table = pl.DataFrame(rows, schema=['fold', 'lambda_io', 'nll_tune', 'n_tune', 'status'], orient='row')
    return float(min(rows, key=lambda r: r[2])[1]), table


def select_kan(panel: Panel, flags: HMMFlags, fold: str,
               Ms: tuple[int, ...] = (8, 12, 16),
               grid: tuple[float, ...] = (0, 0.01, 0.1, 1)) -> tuple[tuple[int, float], pl.DataFrame]:
    rows: list[tuple[str, int, float, float, int, str]] = []
    schema = ['fold', 'M', 'lambda_spl', 'nll_tune', 'n_tune', 'status']
    if not len(internal_split(panel, fold)[1]):
        return (12, 0.01), pl.DataFrame([(fold, 12, 0.01, float('nan'), 0, 'default_empty')], schema=schema, orient='row')
    for M in Ms:
        for lam in grid:
            chosen: HMMFlags = {**flags, 'kan': True, 'M': M, 'lambda_spl': lam}
            score, n = _grid_score(panel, chosen, fold)
            rows.append((fold, M, lam, score, n, 'tuned'))
    best = min(rows, key=lambda r: r[3])
    return (best[1], best[2]), pl.DataFrame(rows, schema=schema, orient='row')
