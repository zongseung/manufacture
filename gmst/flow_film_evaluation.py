import numpy as np
import polars as pl
from typing import TypedDict

from gmst import evaluate, features
from gmst.contracts import FloatArray, Panel, Prediction
from gmst.flow_film_training import FlowIssue, OOFRecord


class DayScore(TypedDict):
    mae: float
    crps: float
    n_slots: int
    peak_error: float
    risk: FloatArray
    event: FloatArray
    errors: FloatArray


def score_prediction(pred: Prediction, panel: Panel, day: int, C: FloatArray) -> DayScore:
    y = panel['Y'][day]
    median = pred['y_median']
    quantiles = pred['q']
    paths = pred['paths']
    if quantiles is None or paths is None:
        raise ValueError('Flow evaluation requires joint paths and quantiles')
    mae, n_slots = evaluate.mae(y, median)
    crps, _ = evaluate.crps(y, quantiles.T)
    maxima = paths.max(axis=1)
    risk = np.where(np.isfinite(C), (maxima[:, None] > C).mean(axis=0), np.nan)
    usable = bool(features.usable_peak(panel)[day])
    true_max = float(np.nanmax(y)) if usable else float('nan')
    event = np.where(np.isfinite(C) & usable, (true_max > C).astype(float), np.nan)
    errors = np.where(np.isfinite(y), np.abs(y - median), np.nan)
    return {'mae': mae, 'crps': crps, 'n_slots': n_slots,
            'peak_error': abs(true_max - pred['M_hat_median']) if usable else float('nan'),
            'risk': risk, 'event': event, 'errors': errors}


def summarize(metrics: pl.DataFrame, required_folds: int = 8) -> tuple[pl.DataFrame, dict[str, object]]:
    candidates = ('B3', 'mlp', 'q', 'j', 'gated_j')
    summary = metrics.group_by('seed', 'candidate').agg(
        pl.col('mae').mean().alias('mae_mean'),
        pl.col('mae').median().alias('mae_median'),
        pl.col('mae').max().alias('mae_worst'),
        pl.col('crps').mean().alias('crps_mean'),
        pl.col('fold').n_unique().alias('n_folds'),
        pl.len().alias('n_rows'),
    ).sort('seed', 'candidate')
    expected = {f'ar{i}' for i in range(1, required_folds + 1)}
    complete = metrics.height == required_folds * 3 * len(candidates)
    for seed in range(3):
        for candidate in candidates:
            group = metrics.filter((pl.col('seed') == seed) & (pl.col('candidate') == candidate))
            if group.height != required_folds or set(group['fold']) != expected:
                complete = False
    by_key = {(int(row['seed']), row['candidate']): row for row in summary.iter_rows(named=True)}
    qualified = complete and all(
        np.isfinite(by_key[seed, 'gated_j']['mae_mean'])
        and by_key[seed, 'gated_j']['mae_mean'] < by_key[seed, control]['mae_mean']
        for seed in range(3) for control in candidates if control != 'gated_j'
    ) and all(
        np.isfinite(by_key[seed, 'gated_j']['crps_mean'])
        and by_key[seed, 'gated_j']['crps_mean'] <= by_key[seed, 'B3']['crps_mean']
        for seed in range(3)
    )
    return summary, {'status': 'development_candidate' if qualified else 'retain_B3',
                     'complete': complete, 'qualified': qualified}


def candidate_metrics(panel: Panel, fold: str, seed: int, candidate: str,
                      outer: list[tuple[FlowIssue, Prediction]],
                      cal: list[tuple[OOFRecord, Prediction]], C: FloatArray,
                      paths: int) -> dict[str, str | int | float]:
    cal_scores = [score_prediction(pred, panel, record.issue.day, C) for record, pred in cal]
    outer_scores = [score_prediction(pred, panel, issue.day, C) for issue, pred in outer]
    n_slots = sum(score['n_slots'] for score in outer_scores)
    result: dict[str, str | int | float] = {
        'fold': fold, 'seed': seed, 'candidate': candidate, 'n_slots': n_slots,
        'n_days': len(outer),
        'mae': sum(score['mae'] * score['n_slots'] for score in outer_scores) / n_slots,
        'crps': sum(score['crps'] * score['n_slots'] for score in outer_scores) / n_slots,
    }
    peaks = np.array([score['peak_error'] for score in outer_scores])
    result['peak_mae'] = float(np.nanmean(peaks)) if np.isfinite(peaks).any() else float('nan')
    switches = np.concatenate([issue.flow.sum(axis=(1, 2)) - np.trace(issue.flow, axis1=1, axis2=2)
                               for issue, _ in outer])
    errors = np.concatenate([score['errors'] for score in outer_scores])
    observed = np.isfinite(errors)
    ordered = np.sort(switches[observed])
    cut = int(np.ceil(0.75 * len(ordered)))
    if 0 < cut < len(ordered) and ordered[cut - 1] < ordered[cut]:
        high = observed & (switches >= ordered[cut])
        low = observed & ~high
        result['mae_high_switch'] = float(errors[high].mean())
        result['mae_low_switch'] = float(errors[low].mean())
    else:
        result['mae_high_switch'] = result['mae_low_switch'] = float('nan')
    result['mean_abs_delta'] = float(np.mean([np.mean(np.abs(pred['y_median'] - issue.base['y_median']))
                                             for issue, pred in outer]))
    for j, name in enumerate(evaluate.CS):
        cal_risk = np.array([score['risk'][j] for score in cal_scores])
        cal_event = np.array([score['event'][j] for score in cal_scores])
        valid_cal = np.isfinite(cal_risk) & np.isfinite(cal_event)
        n_cal = int(valid_cal.sum())
        status = ('default_empty' if n_cal == 0 else
                  'single_class' if np.unique(cal_event[valid_cal]).size == 1 else 'ok')
        ab = evaluate.platt_fit(cal_risk, cal_event, paths)
        risk = np.array([score['risk'][j] for score in outer_scores])
        event = np.array([score['event'][j] for score in outer_scores])
        valid = np.isfinite(risk) & np.isfinite(event)
        calibrated = evaluate.platt_apply(ab, risk, paths)
        result[f'cal_status_{name}'] = status
        result[f'n_cal_{name}'] = n_cal
        result[f'n_brier_{name}'] = int(valid.sum())
        result[f'n_event_{name}'] = int((event[valid] == 1).sum())
        result[f'brier_raw_{name}'] = evaluate.brier(risk, event)
        result[f'brier_platt_{name}'] = evaluate.brier(calibrated, event)
    return result
