from functools import cache

import numpy as np

from gmst import bat, conditional_ar, features
from gmst.conditional_bat import (
    ConditionalConfig,
    ConditionalState,
    conditional_model,
    fit_conditional,
)
from gmst.contracts import Panel
from gmst.evaluate import thresholds
from gmst.run_v3 import _asof


def test_whitening_integrates_missing_slots_and_resets_at_day_boundaries() -> None:
    # Given two partially observed days with a stationary AR covariance.
    y = np.full((2, 96), np.nan)
    y[0, [2, 3, 7]] = 1
    y[1, [0, 5]] = 1
    grid = conditional_ar.ObservationGrid.from_targets(y)
    rho = np.array([.8, .2])
    phi, inv_sd = grid.weights(rho)
    covariance = np.zeros((5, 5))
    for d in range(2):
        rows = np.flatnonzero(grid.day == d)
        t = grid.slot[rows]
        covariance[np.ix_(rows, rows)] = rho[d] ** np.abs(t[:, None] - t) / (1 - rho[d] ** 2)
    # When whitening only the actually observed slots.
    operator = np.diag(inv_sd)
    operator[np.arange(5), grid.previous] -= phi * inv_sd
    # Then covariance is identity, including the gap and each day's first observation.
    np.testing.assert_allclose(operator @ covariance @ operator.T, np.eye(5), atol=1e-12)
    np.testing.assert_allclose(grid.whiten(np.arange(5.), (phi, inv_sd)), operator @ np.arange(5.))


def test_sparse_normal_equations_match_dense_reference() -> None:
    # Given a design with a repeated zero-weight column in the first row.
    columns = np.array([[0, 0, 96], [1, 0, 96], [7, 1, 96]])
    values = np.array([[.5, 0, 2], [1, -.8, 3], [1, -.2, 4]])
    design = conditional_ar.SparseDesign(columns, values)
    weights = np.array([2., 3., 4.])
    target = np.array([1., 2., 5.])
    dense = np.zeros((3, 97))
    np.add.at(dense, (np.arange(3)[:, None], columns), values)
    # When accumulating a weighted precision matrix and right-hand side.
    precision, rhs = design.normal_equations(weights, target)
    # Then the sparse calculation equals the dense reference.
    np.testing.assert_allclose(precision, dense.T @ (weights[:, None] * dense))
    np.testing.assert_allclose(rhs, dense.T @ (weights * target))


@cache
def fitted() -> tuple[Panel, ConditionalState]:
    panel = features.load_panel()
    train = features.role_idx(panel, "f1", "train")
    C, _ = thresholds(panel, train)
    return panel, fit_conditional(panel, train, C, ConditionalConfig(n_iter=160, burn=80))


def test_conditional_fit_separates_noise_and_preserves_joint_peak_contract() -> None:
    panel, state = fitted()
    model = _asof(conditional_model(ConditionalConfig()))
    d = int(features.role_idx(panel, "f1", "val")[0])
    pred = model["predict"](state, panel, d)
    assert np.median(state.sigma_eta[:, 1]) > 2 * np.median(state.sigma_eta[:, 0])
    samples = pred["paths"]
    assert samples is not None and samples.shape == (80, 96) and np.isfinite(samples).all()
    assert pred["M_hat_median"] == np.median(samples.max(axis=1))
    np.testing.assert_array_equal(pred["risk_raw"], (samples.max(axis=1)[:, None] > state.C).mean(axis=0))


def test_forecast_is_deterministic_and_cannot_see_target_day_or_future_power() -> None:
    panel, state = fitted()
    model = _asof(conditional_model(ConditionalConfig()))
    d = int(features.role_idx(panel, "f1", "val")[0])
    changed: Panel = {**panel, "Y": panel["Y"].copy()}
    changed["Y"][d:] = 1e9
    original = model["predict"](state, panel, d)
    forecast = model["predict"](state, changed, d)
    np.testing.assert_array_equal(original["paths"], forecast["paths"])


def test_default_keeps_attention_shape_constant_and_learns_production_response() -> None:
    # Given the existing prior shape and the same training data as the learned model.
    panel = features.load_panel()
    train = features.role_idx(panel, "f1", "train")
    C, _ = thresholds(panel, train)
    config = ConditionalConfig(n_iter=160, burn=80)
    # When the conditional model is refitted with its time response fixed.
    state = fit_conditional(panel, train, C, config)
    # Then only the shape is fixed; production gains and conditional noise remain learned.
    np.testing.assert_array_equal(state.mean["theta"][:, :, :3],
                                  np.broadcast_to(bat.PRIOR_MU[:3], (80, 4, 3)))
    assert (np.ptp(state.mean["w"][:, :, 1:], axis=0) > 0).all()
    assert state.rho.shape == state.sigma_eta.shape == state.sigma_u.shape == (80, 2)
    d = int(features.role_idx(panel, "f1", "val")[0])
    q = np.nan_to_num(panel["X"]["생산량"][d, ::4])
    original = bat.plan_curves(state.mean, panel, d, q)
    changed = bat.plan_curves(state.mean, panel, d, q * 1.1)
    assert np.any(changed > original)
    assert (changed >= original - 1e-10).all()
    assert conditional_model(config)["name"] == "BAT_conditional_gaussian_fixed_attention"
