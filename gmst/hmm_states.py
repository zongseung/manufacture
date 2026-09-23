"""Posthoc state diagnostics; never used to issue a forecast."""
from itertools import combinations
from typing import TypedDict

import numpy as np
import polars as pl
import torch

from gmst.contracts import FloatArray, IntArray, Panel
from gmst.features import cal_flags, internal_split, role_idx
from gmst.hmm_core import CondHMM, backward, forward_logp
from gmst.hmm_forecast import HMMState, issue_centre
from gmst.hmm_training import make_blocks, train_hmm, z_features


class StateSummary(TypedDict):
    delta: FloatArray
    occ: FloatArray
    conclusion: tuple[int, str, int]
    eval_states: IntArray


class StabilityRow(TypedDict):
    seed_a: int
    seed_b: int
    occ_maxdiff: float
    delta_maxdiff: float
    same_order: bool
    same_conclusion: bool
    agreement: float


class StabilityResult(TypedDict):
    rows: list[StabilityRow]
    per_seed: dict[int, StateSummary]
    ok: bool
    collapsed: bool


def posthoc_states(state: HMMState, panel: Panel, d: int) -> tuple[IntArray, IntArray]:
    model = state['model']
    blocks = make_blocks(panel, np.array([d]), issue_centre(state, panel, np.array([d])), state['protocol'], model.W.device, model.W.dtype, kan=model.kan, M=model.M)
    if blocks is None:
        return np.zeros(96, dtype=np.int64), np.zeros(96, dtype=np.int64)
    args = (model, blocks['y'], blocks['obs'], blocks['m'], blocks['opt'], blocks['z'], blocks['u'], blocks['v'])
    with torch.no_grad():
        filt = forward_logp(*args)[1][0, 96:].argmax(-1).cpu().numpy() + 1
        smooth = backward(*args)[0, 96:].argmax(-1).cpu().numpy() + 1
    return filt, smooth


def transition_table(model: CondHMM, panel: Panel, day_idx: IntArray,
                     protocol: str = 'A+') -> pl.DataFrame:
    op, _ = cal_flags(panel, protocol)
    cells = sorted({(int(op[d]), int(panel['dtype'][d])) for d in day_idx})
    rows: list[tuple[int, str, int, int, float]] = []
    for operating, daytype in cells:
        d = next(int(d) for d in day_idx if op[d] == operating and panel['dtype'][d] == daytype)
        view: Panel = {**panel, 'hol': np.zeros_like(panel['hol']), 'X': {**panel['X'], '생산량': np.zeros_like(panel['Y'])}}
        z = torch.as_tensor(z_features(view, protocol, np.array([d]), kan=model.kan, M=model.M), dtype=model.W.dtype, device=model.W.device)
        with torch.no_grad():
            high = model.trans(z)[0, :, :, -1].cpu().numpy().reshape(24, 4, model.K).mean(1)
        for hour in range(24):
            for state in range(model.K):
                rows.append((operating, ('wk', 'sat', 'sun')[daytype], hour, state+1, float(high[hour, state])))
    return pl.DataFrame(rows, schema=['op', 'daytype', 'hour', 'from_state', 'p_to_high'], orient='row')


def conclusion(table: pl.DataFrame, K: int) -> tuple[int, str, int]:
    best = table.filter(pl.col('from_state') == max(K-1, 1)).sort('p_to_high', descending=True, maintain_order=True)
    return int(best['op'][0]), str(best['daytype'][0]), int(best['hour'][0])


def state_summary(model: CondHMM, panel: Panel, train_idx: IntArray,
                  eval_idx: IntArray, m: FloatArray, protocol: str = 'A+') -> StateSummary:
    def marginals(days: IntArray) -> tuple[FloatArray, IntArray]:
        b = make_blocks(panel, days, m, protocol, model.W.device, model.W.dtype, kan=model.kan, M=model.M)
        if b is None:
            return np.zeros((0, model.K)), np.array([], dtype=np.int64)
        with torch.no_grad():
            gamma = backward(model, b['y'], b['obs'], b['m'], b['opt'], b['z'], b['u'], b['v'])[:, 96:]
        observed = gamma[b['obs'][:, 96:]].cpu().numpy()
        return observed, observed.argmax(-1) + 1
    train, _ = marginals(train_idx)
    return {'delta': model.delta().detach().cpu().numpy().astype(float),
            'occ': train.mean(0) if len(train) else np.zeros(model.K),
            'conclusion': conclusion(transition_table(model, panel, train_idx, protocol), model.K),
            'eval_states': marginals(eval_idx)[1]}


def stability_rows(summaries: dict[int, StateSummary]) -> list[StabilityRow]:
    rows: list[StabilityRow] = []
    for a, b in combinations(summaries, 2):
        left, right = summaries[a], summaries[b]
        agreement = float(np.mean(left['eval_states'] == right['eval_states'])) if len(left['eval_states']) else float('nan')
        rows.append({'seed_a': a, 'seed_b': b,
            'occ_maxdiff': float(np.abs(left['occ']-right['occ']).max()),
            'delta_maxdiff': float(np.abs(left['delta']-right['delta']).max()),
            'same_order': bool(np.all(np.diff(left['delta'], axis=0)>0) and np.all(np.diff(right['delta'], axis=0)>0)),
            'same_conclusion': left['conclusion'] == right['conclusion'], 'agreement': agreement})
    return rows


def state_stability(panel: Panel, fold: str = 'f1', seeds: tuple[int, ...] = (0, 1, 2),
                    m: FloatArray | None = None, tau: float = 10, half_life: int = 60,
                    protocol: str = 'A+', *, max_epochs: int = 300, patience: int = 20,
                    device: str | torch.device = 'cpu', occ_floor: bool = False) -> StabilityResult:
    from gmst.backbone import fold_backbone
    tr, ev = role_idx(panel, fold, 'train'), role_idx(panel, fold, 'val')
    fit_idx, tune, _ = internal_split(panel, fold)
    m_fit = fold_backbone(panel, fit_idx, tau, half_life, protocol) if m is None else m
    means = fold_backbone(panel, tr, tau, half_life, protocol) if m is None else m
    summaries: dict[int, StateSummary] = {}
    for seed in seeds:
        best = min(20, max_epochs)
        if len(tune):
            _, best, _ = train_hmm(panel, fit_idx, tune, m_fit, protocol, seed=seed, max_epochs=max_epochs, patience=patience, device=device, occ_floor=occ_floor)
        model, _, _ = train_hmm(panel, tr, np.array([], dtype=np.int64), means, protocol, seed=seed, epochs=best, device=device, occ_floor=occ_floor)
        summaries[seed] = state_summary(model, panel, tr, ev, means, protocol)
    rows = stability_rows(summaries)
    return {'rows': rows, 'per_seed': summaries,
            'ok': all(r['occ_maxdiff'] < .05 and r['delta_maxdiff'] < 5 and r['same_order'] and r['same_conclusion'] for r in rows),
            'collapsed': any(float(s['occ'].min()) < .02 for s in summaries.values())}
