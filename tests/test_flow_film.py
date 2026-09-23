import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import polars as pl
import pytest
import torch

from gmst import RESULTS, ROOT, hmm
from gmst.hmm_forecast import day_tensors, filter_last, issue_centre
from gmst.flow_film import FlowFiLM, issue_flow, shift_prediction
from gmst.flow_film_experiment import (
    DiagnosticConfig, DiagnosticRun, diagnose_fold, diagnostic_decision, run_diagnostic,
    run_experiment, score_prediction, summarize,
)
from gmst.flow_film_training import FitConfig, fit_head, inner_oof
from test_hmm import synthetic_panel


def test_issue_flow_has_the_correct_marginals_and_no_ar() -> None:
    panel = synthetic_panel()
    b3 = hmm.hmm_model('B3', max_epochs=2, N=16, seed=0)
    state = b3['fit'](panel, 'f1', np.array([20.0, 30.0, 40.0]))
    day = 25

    flow = issue_flow(state, panel, day)
    m = issue_centre(state, panel, np.array([day]))
    q0, _ = filter_last(state['model'], panel, day, m, 'A+')
    z = day_tensors(state['model'], panel, np.array([day]), m, 'A+')[0]
    transition = state['model'].trans(z)[0].detach().cpu().numpy()

    assert torch.count_nonzero(state['model'].phi()) == 0
    assert flow.shape == (96, 3, 3)
    assert np.all(flow >= 0)
    np.testing.assert_allclose(flow.sum(axis=(1, 2)), 1, atol=1e-6)
    np.testing.assert_allclose(flow[0].sum(axis=1), q0.detach().cpu().numpy(), atol=1e-6)
    np.testing.assert_allclose(flow[1:].sum(axis=2), flow[:-1].sum(axis=1), atol=1e-6)
    np.testing.assert_allclose(flow, flow.sum(axis=2)[..., None] * transition, atol=1e-6)


def test_issue_flow_ignores_current_target_and_handles_missing_predecessor() -> None:
    panel = synthetic_panel()
    b3 = hmm.hmm_model('B3', max_epochs=2, N=16, seed=0)
    state = b3['fit'](panel, 'f1', np.array([20.0, 30.0, 40.0]))
    baseline = issue_flow(state, panel, 25)
    changed = {**panel, 'Y': panel['Y'].copy()}
    changed['Y'][25] = 9999.0
    changed_flow = issue_flow(state, changed, 25)
    changed['Y'][24] = np.nan
    missing_previous = issue_flow(state, changed, 25)

    np.testing.assert_array_equal(changed_flow, baseline)
    assert np.isfinite(missing_previous).all()
    np.testing.assert_allclose(missing_previous.sum(axis=(1, 2)), 1, atol=1e-6)


@pytest.mark.parametrize('mode', ['mlp', 'q', 'j', 'gated_j'])
def test_flow_film_starts_at_b3_and_gate_zeros_without_switch(mode: str) -> None:
    head = FlowFiLM(mode, 10.0)
    x = torch.ones((96, 15))
    flow = torch.eye(3).expand(96, 3, 3) / 3
    initial = head(x, flow)
    with torch.no_grad():
        head.output.bias.fill_(1.0)
    after_bias = head(x, flow)

    assert torch.count_nonzero(initial) == 0
    if mode == 'gated_j':
        assert torch.count_nonzero(after_bias) == 0
    else:
        assert torch.all(after_bias > 0)


def test_shift_prediction_recomputes_joint_path_outputs() -> None:
    paths = np.vstack((np.zeros(96), np.ones(96)))
    baseline = {'paths': paths, 'y_mean': paths.mean(0), 'y_median': np.median(paths, 0),
                'q': np.quantile(paths, np.arange(1, 20) / 20, axis=0),
                'M_hat_median': 0.5, 'M_hat_mean': 0.5, 'peak_time_mode': 0,
                'risk_raw': np.zeros(3)}
    delta = np.arange(96, dtype=float) / 95

    shifted = shift_prediction(baseline, delta, np.array([0.5, 1.5, 2.5]))

    np.testing.assert_array_equal(shifted['paths'], paths + delta[None, :])
    np.testing.assert_allclose(shifted['y_mean'], paths.mean(0) + delta)
    assert shifted['q'].shape == (19, 96)
    assert np.all(np.diff(shifted['q'], axis=0) >= 0)
    assert shifted['M_hat_median'] == 1.5
    assert shifted['peak_time_mode'] == 95
    np.testing.assert_array_equal(shifted['risk_raw'], np.array([1.0, 0.5, 0.0]))
    np.testing.assert_array_equal(baseline['paths'], paths)


def test_diagnostic_gate_requires_six_positive_windows_per_seed() -> None:
    rows = pl.DataFrame([
        {'fold': f'ar{fold}', 'seed': seed, 's': float(slot), 'abs_err': float(slot), 'observed': True}
        for fold in range(1, 9) for seed in range(3) for slot in range(4)
    ])
    supported = diagnostic_decision(rows)
    weaker = rows.with_columns(pl.when((pl.col('seed') == 2) & (pl.col('fold').is_in(['ar6', 'ar7', 'ar8'])))
        .then(3 - pl.col('abs_err')).otherwise(pl.col('abs_err')).alias('abs_err'))
    rejected = diagnostic_decision(weaker)

    assert supported['status'] == 'supported'
    assert supported['positive_folds'] == {'0': 8, '1': 8, '2': 8}
    assert rejected['status'] == 'transition_hypothesis_unsupported'
    assert rejected['positive_folds']['2'] == 5


def test_diagnostic_gate_rejects_tied_quartile_boundary() -> None:
    rows = pl.DataFrame([
        {'fold': f'ar{fold}', 'seed': seed, 's': float(slot > 0), 'abs_err': float(slot), 'observed': True}
        for fold in range(1, 9) for seed in range(3) for slot in range(4)
    ])
    verdict = diagnostic_decision(rows)

    assert verdict['status'] == 'transition_hypothesis_unsupported'
    assert verdict['informative_folds'] == {'0': 0, '1': 0, '2': 0}


def test_diagnosis_uses_only_prior_y_for_issued_flow() -> None:
    panel = synthetic_panel()
    config = DiagnosticConfig(seed=0, epochs=2, paths=16)
    baseline = diagnose_fold(panel, 'f1', config)
    changed = {**panel, 'Y': panel['Y'].copy()}
    changed['Y'][25] = 9999.0
    altered = diagnose_fold(changed, 'f1', config)

    assert baseline.height == 3 * 96
    assert set(baseline['fold']) == {'f1'}
    assert all(baseline['observed'])
    first_date = panel['dates'][25].isoformat()
    before = baseline.filter(pl.col('date') == first_date)
    after = altered.filter(pl.col('date') == first_date)
    for column in ('s', 'y_median', 'q0', 'q1', 'q2', 'J00', 'J01', 'J02', 'P00', 'P01', 'P02'):
        np.testing.assert_array_equal(before[column].to_numpy(), after[column].to_numpy())
    assert not np.array_equal(before['abs_err'].to_numpy(), after['abs_err'].to_numpy())


def test_diagnostic_smoke_writes_isolated_artifacts(tmp_path: Path) -> None:
    frozen = RESULTS / 'selection.json'
    before = hashlib.sha256(frozen.read_bytes()).hexdigest()
    out = tmp_path / 'flow-smoke'

    run_diagnostic(out, DiagnosticRun(smoke=True, epochs=2, paths=16))

    rows = pl.read_csv(out / 'diagnostic.csv')
    decision = json.loads((out / 'decision.json').read_text())
    assert rows.height == 14 * 96
    assert set(rows['fold']) == {'ar8'}
    assert set(rows['seed']) == {0}
    assert decision['status'] == 'smoke_only'
    assert hashlib.sha256(frozen.read_bytes()).hexdigest() == before


def test_diagnostic_refuses_existing_and_protected_outputs(tmp_path: Path) -> None:
    config = DiagnosticRun(smoke=True, epochs=2, paths=16)
    with pytest.raises(ValueError, match='protected'):
        run_diagnostic(RESULTS / 'new-flow', config)
    with pytest.raises(ValueError, match='protected'):
        run_diagnostic(ROOT / 'results_ar' / 'new-flow', config)
    existing = tmp_path / 'existing'
    existing.mkdir()
    with pytest.raises(ValueError, match='exists'):
        run_diagnostic(existing, config)


@pytest.mark.parametrize('flag', ['--epochs', '--paths'])
def test_cli_rejects_zero_budget_before_training(tmp_path: Path, flag: str) -> None:
    result = subprocess.run(
        [sys.executable, '-m', 'gmst.flow_film_experiment', '--diagnostic', flag, '0',
         '--out', str(tmp_path / 'never')],
        cwd=ROOT, capture_output=True, text=True, timeout=5, check=False,
    )

    assert result.returncode == 2
    assert 'positive' in result.stderr
    assert not (tmp_path / 'never').exists()


def test_inner_oof_is_strictly_prior_and_cap_uses_training_rows() -> None:
    panel = synthetic_panel(80)
    panel['days'] = panel['days'].with_columns(pl.Series('event_id', [f'E{i:03d}' for i in range(80)]))
    config = FitConfig(seed=0, epochs=2, paths=8)

    blocks = inner_oof(panel, 'f1', config)
    fitted = fit_head('gated_j', blocks, config)

    assert len(blocks.train) == len(blocks.tune) == len(blocks.cal) == 14
    targets = [record.issue.day for group in (blocks.train, blocks.tune, blocks.cal) for record in group]
    assert len(set(targets)) == 42
    assert max(targets) < min(np.flatnonzero((panel['days']['f1'] == 'val').to_numpy()))
    assert all(record.issue.fit_end < record.issue.day for group in (blocks.train, blocks.tune, blocks.cal)
               for record in group)
    errors = np.concatenate([np.abs(record.y - record.issue.base['y_median'])
                             for record in blocks.train])
    expected_cap = float(np.quantile(errors[np.isfinite(errors)], 0.9))
    assert fitted.head.scale == pytest.approx(expected_cap)


def test_summary_requires_every_seed_to_beat_all_controls() -> None:
    rows = pl.DataFrame([
        {'fold': f'ar{fold}', 'seed': seed, 'candidate': candidate, 'mae': mae, 'crps': crps}
        for fold in range(1, 9) for seed in range(3)
        for candidate, mae, crps in [('B3', 10.0, 8.0), ('mlp', 9.8, 8.0),
                                     ('q', 9.7, 8.0), ('j', 9.6, 8.0), ('gated_j', 9.5, 7.9)]
    ])
    summary, decision = summarize(rows)
    tied = rows.with_columns(pl.when((pl.col('seed') == 2) & (pl.col('candidate') == 'gated_j'))
        .then(9.6).otherwise(pl.col('mae')).alias('mae'))
    missing = rows.filter(~((pl.col('seed') == 2) & (pl.col('fold') == 'ar8')))

    assert decision['status'] == 'development_candidate'
    assert summary.filter(pl.col('candidate') == 'gated_j').height == 3
    assert summarize(tied)[1]['status'] == 'retain_B3'
    assert summarize(missing)[1]['status'] == 'retain_B3'


def test_score_prediction_masks_missing_slots_but_preserves_full_paths() -> None:
    panel = synthetic_panel()
    b3 = hmm.hmm_model('B3', max_epochs=2, N=16, seed=0)
    state = b3['fit'](panel, 'f1', np.array([20.0, 30.0, 40.0]))
    from gmst.flow_film_training import issue_day
    issue = issue_day(state, panel, 25, 16)
    changed = {**panel, 'Y': panel['Y'].copy()}
    changed['Y'][25, 0] = np.nan

    result = score_prediction(issue.base, changed, 25, np.array([20.0, 30.0, 40.0]))

    assert result['n_slots'] == 95
    assert issue.base['paths'].shape == (16, 96)
    assert all(0 <= value <= 1 for value in result['risk'])


def test_neural_smoke_writes_five_candidates_without_adoption(tmp_path: Path) -> None:
    frozen = RESULTS / 'selection.json'
    before = hashlib.sha256(frozen.read_bytes()).hexdigest()
    out = tmp_path / 'neural-smoke'

    run_experiment(out, DiagnosticRun(smoke=True, epochs=2, paths=8))

    metrics = pl.read_csv(out / 'metrics.csv')
    decision = json.loads((out / 'decision.json').read_text())
    assert metrics.height == 5
    assert set(metrics['candidate']) == {'B3', 'mlp', 'q', 'j', 'gated_j'}
    assert set(metrics['fold']) == {'ar8'}
    assert set(metrics['seed']) == {0}
    assert decision['status'] == 'smoke_only'
    assert hashlib.sha256(frozen.read_bytes()).hexdigest() == before
