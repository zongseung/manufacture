"""Origin and fit perturbations test the real model paths, not canned predictions."""
from copy import deepcopy

import numpy as np
import pytest
from test_hmm import synthetic_panel

from gmst import hmm
from gmst.features import internal_split, role_idx


@pytest.mark.parametrize('kind,decoder,kan', [('B2','constant',False), ('B3','constant',False), ('B4','constant',False), ('B4','io',False), ('B4','constant',True)])
def test_forecast_ignores_current_and_future_target_and_exogenous_values(kind, decoder, kan):
    # Given a model fitted solely on the training prefix.
    panel = synthetic_panel()
    factory = hmm.hmm_model(kind, decoder=decoder, kan=kan, max_epochs=1, N=50, device='cpu')
    state = factory['fit'](panel, 'f1', np.array([10, 15, 20.]))
    d = int(role_idx(panel, 'f1', 'val')[0])
    altered = deepcopy(panel)
    altered['Y'][d:] = 500
    for value in altered['X'].values():
        value[d:] = 300
    # When issuing at the same origin from perturbed post-origin observations.
    before, after = factory['predict'](state, panel, d), factory['predict'](state, altered, d)
    # Then complete MC paths are identical.
    np.testing.assert_array_equal(before['paths'], after['paths'])


def test_fitting_and_calibration_prefix_are_outer_label_invariant():
    # Given outer observations changed after the last training day.
    panel = synthetic_panel()
    altered = deepcopy(panel)
    tr = role_idx(panel, 'f1', 'train')
    altered['Y'][tr[-1]+1:] = 500
    factory = hmm.hmm_model(max_epochs=2, N=20, device='cpu')
    # When fitting both panels.
    original = factory['fit'](panel, 'f1', np.array([10, 15, 20.]))
    changed = factory['fit'](altered, 'f1', np.array([10, 15, 20.]))
    # Then fitted parameters agree and calibration model ends before its first target.
    for key, value in original['model'].state_dict().items():
        np.testing.assert_array_equal(value.detach().numpy(), changed['model'].state_dict()[key].detach().numpy())
    cal = internal_split(panel, 'f1')[2]
    assert original['inner_state']['train_idx'].max() < cal.min()
    assert original['best_epoch'] == changed['best_epoch']
    assert 'val_nll' not in original


def test_protocol_b_uses_only_current_allowed_production():
    # Given a conditional production model at an issue day.
    p = synthetic_panel()
    d = 3
    model = hmm.CondHMM(dz=16).double()
    m, C = np.zeros_like(p['Y']), np.array([20., 30., 40.])
    future, current = deepcopy(p), deepcopy(p)
    future['X']['생산량'][d+1:] = 10000
    current['X']['생산량'][d] = 10000
    # When forecasting with allowed current production or only future production changed.
    reference = hmm.forecast(model, p, d, m, C, 'B', N=500)
    same = hmm.forecast(model, future, d, m, C, 'B', N=500)
    changed = hmm.forecast(model, current, d, m, C, 'B', N=500)
    # Then the current plan matters and future plans do not.
    np.testing.assert_array_equal(reference['paths'], same['paths'])
    assert not np.array_equal(reference['paths'], changed['paths'])


@pytest.mark.parametrize('kan', [False, True])
def test_inner_grid_selection_ignores_outer_labels(kan):
    # Given tiny registered grids and a changed validation suffix.
    p = synthetic_panel()
    changed = deepcopy(p)
    changed['Y'][24:] = 1000
    flags: hmm.HMMFlags = {'max_epochs': 1, 'device': 'cpu', 'K': 2}
    # When choosing on the training-only tuning window.
    if kan:
        a, table_a = hmm.select_kan(p, flags, 'f1', Ms=(8,), grid=(0., .01))
        b, table_b = hmm.select_kan(changed, flags, 'f1', Ms=(8,), grid=(0., .01))
    else:
        a, table_a = hmm.select_io_lambda(p, flags, 'f1', grid=(0., .01))
        b, table_b = hmm.select_io_lambda(changed, flags, 'f1', grid=(0., .01))
    # Then candidate scores and selection are invariant.
    assert a == b
    np.testing.assert_array_equal(table_a['nll_tune'].to_numpy(), table_b['nll_tune'].to_numpy())


def test_hybrid_fit_and_issue_forecasts_are_causal():
    # Given the hybrid model and perturbed observations after its training end.
    p = synthetic_panel()
    altered = deepcopy(p)
    altered['Y'][24:] = 1000
    for values in altered['X'].values():
        values[24:] = 1000
    factory = hmm.hmm_model(emission_source='b1', rounds=2, max_epochs=1, N=20, device='cpu')
    # When fitting and issuing the same first outer date.
    first = factory['fit'](p, 'f1', np.array([10., 15., 20.]))
    second = factory['fit'](altered, 'f1', np.array([10., 15., 20.]))
    d = int(role_idx(p, 'f1', 'val')[0])
    forecast_panel = deepcopy(p)
    forecast_panel['Y'][d:] = 500
    for values in forecast_panel['X'].values():
        values[d:] = 500
    a, b = factory['predict'](first, p, d), factory['predict'](first, forecast_panel, d)
    # Then centre matrices and fitted parameters ignore all outer values, and paths ignore post-origin values.
    np.testing.assert_array_equal(first['m'], second['m'])
    for key, value in first['model'].state_dict().items():
        np.testing.assert_array_equal(value.detach().numpy(), second['model'].state_dict()[key].detach().numpy())
    np.testing.assert_array_equal(a['paths'], b['paths'])
