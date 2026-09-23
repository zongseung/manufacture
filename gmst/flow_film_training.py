from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import polars as pl
import torch

from gmst import backbone, evaluate, features, hmm, splits
from gmst.contracts import FloatArray, Panel, Prediction
from gmst.flow_film import FlowFiLM, FlowMode, issue_flow, shift_prediction
from gmst.hmm_forecast import HMMState, issue_centre
from gmst.hmm_training import z_features


@dataclass(frozen=True, slots=True)
class FitConfig:
    seed: int
    epochs: int
    paths: int


@dataclass(frozen=True, slots=True)
class FlowIssue:
    day: int
    fit_end: int
    x: FloatArray
    flow: FloatArray
    base: Prediction


@dataclass(frozen=True, slots=True)
class OOFRecord:
    issue: FlowIssue
    y: FloatArray


@dataclass(frozen=True, slots=True)
class OOFBlocks:
    train: list[OOFRecord]
    tune: list[OOFRecord]
    cal: list[OOFRecord]
    C: FloatArray


@dataclass(frozen=True, slots=True)
class FlowFit:
    head: FlowFiLM
    x_mean: FloatArray
    x_sd: FloatArray

    def delta(self, issue: FlowIssue) -> FloatArray:
        x = torch.as_tensor((issue.x - self.x_mean) / self.x_sd, dtype=torch.float32)
        flow = torch.as_tensor(issue.flow, dtype=torch.float32)
        with torch.no_grad():
            return self.head(x, flow).numpy().astype(np.float64)

    def predict(self, issue: FlowIssue, C: FloatArray) -> Prediction:
        return shift_prediction(issue.base, self.delta(issue), C)


def issue_day(state: HMMState, panel: Panel, day: int, paths: int) -> FlowIssue:
    m = issue_centre(state, panel, np.array([day], dtype=np.int64))
    base = hmm.forecast(state['model'], panel, day, m, state['C'], state['protocol'],
                        N=paths, device=state['device'])
    z = z_features(panel, 'A+', np.array([day], dtype=np.int64))[0]
    x = np.column_stack((z, base['y_median']))
    return FlowIssue(day, int(state['train_idx'].max()), x, issue_flow(state, panel, day), base)


def inner_oof(panel: Panel, outer_fold: str, config: FitConfig) -> OOFBlocks:
    eligible = features.role_idx(panel, outer_fold, 'train')
    eligible = eligible[np.isfinite(panel['Y'][eligible]).any(axis=1)]
    if len(eligible) < 42:
        raise ValueError('Flow experiment needs 42 usable prior training days')
    blocks = np.array_split(eligible[-42:], 3)
    C, _ = evaluate.thresholds(panel, eligible[eligible < blocks[2][0]])
    events = panel['days']['event_id'].to_list()
    outer_train = set(features.role_idx(panel, outer_fold, 'train'))
    records: list[list[OOFRecord]] = []
    for block in blocks:
        roles = splits.fold_roles(panel['dates'], events, panel['dates'][int(block[0])],
                                  panel['dates'][int(block[-1])], 'val')
        roles = [role if i in outer_train else '' for i, role in enumerate(roles)]
        inner_panel = {**panel, 'days': panel['days'].with_columns(pl.Series('flow_inner', roles))}
        tau, half_life, _ = backbone.select_tau(inner_panel, 'flow_inner')
        b3 = hmm.hmm_model('B3', K=3, tau=tau, half_life=half_life,
                           max_epochs=config.epochs, N=config.paths, seed=config.seed)
        inner_C, _ = evaluate.thresholds(inner_panel, features.role_idx(inner_panel, 'flow_inner', 'train'))
        state = b3['fit'](inner_panel, 'flow_inner', inner_C)
        records.append([OOFRecord(issue_day(state, inner_panel, int(day), config.paths),
                                  panel['Y'][day].copy()) for day in block])
    return OOFBlocks(records[0], records[1], records[2], C)


def _training_arrays(records: list[OOFRecord]) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    x = np.vstack([record.issue.x for record in records])
    flow = np.vstack([record.issue.flow for record in records])
    y = np.concatenate([record.y for record in records])
    base = np.concatenate([record.issue.base['y_median'] for record in records])
    observed = np.isfinite(y) & np.isfinite(base)
    return x[observed], flow[observed], y[observed], base[observed]


def fit_head(mode: FlowMode, blocks: OOFBlocks, config: FitConfig) -> FlowFit:
    train_x, train_flow, train_y, train_base = _training_arrays(blocks.train)
    tune_x, tune_flow, tune_y, tune_base = _training_arrays(blocks.tune)
    scale = float(np.quantile(np.abs(train_y - train_base), 0.9))
    x_mean = train_x.mean(axis=0)
    x_sd = np.maximum(train_x.std(axis=0), 1.0)
    x_train = torch.as_tensor((train_x - x_mean) / x_sd, dtype=torch.float32)
    x_tune = torch.as_tensor((tune_x - x_mean) / x_sd, dtype=torch.float32)
    f_train = torch.as_tensor(train_flow, dtype=torch.float32)
    f_tune = torch.as_tensor(tune_flow, dtype=torch.float32)
    y_train = torch.as_tensor(train_y - train_base, dtype=torch.float32)
    y_tune = torch.as_tensor(tune_y - tune_base, dtype=torch.float32)
    torch.manual_seed(config.seed)
    head = FlowFiLM(mode, scale)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    best = float('inf')
    best_weights = deepcopy(head.state_dict())
    stalled = 0
    for _ in range(min(config.epochs, 50)):
        optimizer.zero_grad()
        torch.abs(y_train - head(x_train, f_train)).mean().backward()
        optimizer.step()
        with torch.no_grad():
            tune_loss = float(torch.abs(y_tune - head(x_tune, f_tune)).mean())
        if tune_loss < best:
            best, best_weights, stalled = tune_loss, deepcopy(head.state_dict()), 0
        else:
            stalled += 1
            if stalled >= 8:
                break
    head.load_state_dict(best_weights)
    head.eval()
    return FlowFit(head, x_mean, x_sd)
