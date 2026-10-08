from dataclasses import replace
from functools import cache

import numpy as np
import pytest

from gmst import bat, conditional_ar, features
from gmst.benchmark_models import b0_kind_ref
from gmst.conditional_bat import (
    ConditionalConfig,
    ConditionalState,
    conditional_model,
    fit_conditional,
    raw_paths,
    slot_states,
    t_precision_weights,
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


def test_transition_variance_reduces_to_constant_formula_and_whitens_heteroscedastic_ar() -> None:
    y = np.full((2, 96), np.nan)
    y[0, [2, 3, 7, 8]] = 1
    y[1, [0, 5, 6]] = 1
    grid = conditional_ar.ObservationGrid.from_targets(y)
    rho = np.array([.8, .3])
    phi, inv_sd = grid.weights(rho)
    # Given equal innovation variances, the general sum equals σ²(1−ρ^{2s})/(1−ρ²) (first obs σ²/(1−ρ²)).
    np.testing.assert_allclose(grid.transition_variance(rho, np.full((2, 96), 2.)), 2. / inv_sd ** 2, rtol=1e-12)
    # Given slot-varying variances, e_t = ρ e_{t−1} + √v_t η_t started stationary at each day's first observation.
    var = np.random.default_rng(0).uniform(.5, 3., (2, 96))
    covariance = np.zeros((7, 7))
    for d in range(2):
        rows = np.flatnonzero(grid.day == d)
        t0 = grid.slot[rows[0]]
        L = np.zeros((96 - t0, 96 - t0))
        L[0, 0] = np.sqrt(var[d, t0] / (1 - rho[d] ** 2))
        for i in range(1, 96 - t0):
            L[i] = rho[d] * L[i - 1]
            L[i, i] = np.sqrt(var[d, t0 + i])
        block = (L @ L.T)[np.ix_(grid.slot[rows] - t0, grid.slot[rows] - t0)]
        covariance[np.ix_(rows, rows)] = block
    # When whitening with the heteroscedastic gap variance, the covariance is identity.
    operator = np.diag(1 / np.sqrt(grid.transition_variance(rho, var)))
    operator[np.arange(7), grid.previous] -= phi * np.diag(operator)
    np.testing.assert_allclose(operator @ covariance @ operator.T, np.eye(7), atol=1e-12)


def test_slot_states_mark_production_and_transitions_within_four_slots() -> None:
    q = np.zeros((2, 24))
    q[0, 10:14] = 5.  # producing slots 40..55
    states = slot_states(q)
    assert states.shape == (2, 96)
    assert (states[1] == 0).all()  # non-operating day is one state
    expected = np.full(96, 2)
    expected[40:56] = 1
    expected[36:44] = 3  # flag changes between 39 and 40: slots within ±4 of the switch
    expected[52:60] = 3  # switch between 55 and 56
    np.testing.assert_array_equal(states[0], expected)
    edge = np.zeros((1, 24))
    edge[0, 0] = 1.  # producing slots 0..3, switch at 3|4 clipped at the day start
    np.testing.assert_array_equal(slot_states(edge)[0, :9], [3] * 8 + [2])


def test_noise_state_slot_fits_state_innovations_and_uses_them_in_paths() -> None:
    panel = features.load_panel()
    train = features.role_idx(panel, "f1", "train")
    C, _ = thresholds(panel, train)
    Y = panel["Y"].copy()
    Y[train[::3], 1::2] = np.nan  # gaps that straddle state changes exercise the mixed-span correction
    config = ConditionalConfig(n_iter=40, burn=20, noise_state="slot")
    state = fit_conditional({**panel, "Y": Y}, train, C, config)
    # Given noise_state="slot": σ_η per slot state, ρ and σ_u on the op split.
    assert state.sigma_eta.shape == (20, 4) and state.rho.shape == state.sigma_u.shape == (20, 2)
    assert state.noise_state == "slot" and np.isfinite(state.sigma_eta).all() and (state.sigma_eta > 0).all()
    assert conditional_model(config)["name"] == "BAT_conditional_gaussian_fixed_attention_noiseslot"
    val = features.role_idx(panel, "f1", "val")
    d = next(int(d) for d in val if (slot_states(panel["X"]["생산량"][d, ::4][None])[0] == 3).any())
    first = int(np.argmax(slot_states(panel["X"]["생산량"][d, ::4][None])[0] == 3))
    # When only the transition-state σ_η is inflated, slots before the first transition are untouched.
    boosted_eta = state.sigma_eta.copy()
    boosted_eta[:, 3] *= 50
    base, boosted = raw_paths(state, panel, d), raw_paths(replace(state, sigma_eta=boosted_eta), panel, d)
    assert np.isfinite(base).all() and base.shape == (20, 96)
    np.testing.assert_array_equal(boosted[:, :first], base[:, :first])
    assert np.var(boosted[:, first]) > 100 * np.var(base[:, first])


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
    assert conditional_model(replace(config, ref="kind"))["name"] == "BAT_conditional_gaussian_fixed_attention_kindref"


def test_kind_reference_is_past_only_and_matches_operating_kind() -> None:
    panel = features.load_panel()
    days = features.role_idx(panel, "f1", "train")
    kind = bat.kinds(panel, np.arange(len(panel["dates"])))
    means = np.zeros((4, 96))
    ref = bat._ref(panel, days, kind[days], means, 1., "kind")
    for i, d in enumerate(days):
        # Hiding the target day and everything after it cannot change its reference curve.
        hidden: Panel = {**panel, "Y": panel["Y"].copy()}
        hidden["Y"][d:] = 1e9
        np.testing.assert_array_equal(bat._ref(hidden, days[i:i + 1], kind[[d]], means, 1., "kind")[0], ref[i])
        r, _ = b0_kind_ref(panel, int(d))
        np.testing.assert_array_equal(ref[i], means[kind[d]] if r is None else panel["Y"][r])
        complete = np.isfinite(panel["Y"][:d]).all(axis=1)
        if (complete & (kind[:d] == kind[d])).any():
            assert r is not None and r < d and kind[r] == kind[d]
    with pytest.raises(ValueError):
        ConditionalConfig(ref="weekday")


def test_op_reference_reproduces_previous_predictions() -> None:
    # Golden values from the pre-`ref` implementation (B0′ op/daytype reference).
    panel = features.load_panel()
    train = features.role_idx(panel, "f1", "train")
    C, _ = thresholds(panel, train)
    config = ConditionalConfig(n_iter=40, burn=20, ref="op")
    state = fit_conditional(panel, train, C, config)
    d = int(features.role_idx(panel, "f1", "val")[0])
    paths = _asof(conditional_model(config))["predict"](state, panel, d)["paths"]
    assert paths is not None
    np.testing.assert_allclose([paths.sum(), paths[0, 50], paths[-1].max()],
                               [253225.45400727852, 104.36288253742437, 213.39622109130798], rtol=1e-12)


def test_noise_state_kind_fits_four_groups_and_raw_paths_matches_day_kind() -> None:
    panel = features.load_panel()
    train = features.role_idx(panel, "f1", "train")
    C, _ = thresholds(panel, train)
    config = ConditionalConfig(n_iter=60, burn=30, noise_state="kind")
    state = fit_conditional(panel, train, C, config)
    # Given noise_state="kind": four noise states (one per operating kind), not two.
    assert state.rho.shape == state.sigma_eta.shape == state.sigma_u.shape == (30, 4)
    assert conditional_model(config)["name"] == "BAT_conditional_gaussian_fixed_attention_noisekind"
    d = int(features.role_idx(panel, "f1", "val")[0])
    q = np.nan_to_num(panel["X"]["생산량"][d, ::4])
    g = bat.kind_of(q)
    other = (g + 1) % 4
    # When only the day's own kind's sigma_u is inflated, raw_paths must pick that state up;
    # inflating any other kind's sigma_u must leave this day's path untouched (same rng draws).
    zeroed = replace(state, sigma_u=np.zeros_like(state.sigma_u))
    spike = np.zeros(4)
    spike[g] = 50.
    boosted = replace(zeroed, sigma_u=np.broadcast_to(spike, zeroed.sigma_u.shape).copy())
    spike_other = np.zeros(4)
    spike_other[other] = 50.
    unaffected = replace(zeroed, sigma_u=np.broadcast_to(spike_other, zeroed.sigma_u.shape).copy())
    baseline = raw_paths(zeroed, panel, d)
    np.testing.assert_array_equal(raw_paths(unaffected, panel, d), baseline)
    assert np.var(raw_paths(boosted, panel, d), axis=0).mean() > 100 * max(np.var(baseline, axis=0).mean(), 1e-12)


def test_noise_state_rejects_invalid_or_unsupported_values() -> None:
    with pytest.raises(ValueError):
        ConditionalConfig(noise_state="weekday")
    with pytest.raises(ValueError):
        ConditionalConfig(conditional=False, noise_state="slot")
    with pytest.raises(ValueError):
        ConditionalConfig(conditional=False, noise_state="kind")
    ConditionalConfig(conditional=False, noise_state="op")  # still allowed


def test_t_precision_weights_draw_the_conjugate_gamma() -> None:
    # Given standardized innovations z, time weights ω and ν: λ | z ~ Gamma((ν+ω)/2, rate (ν+ω·z²)/2).
    z, omega, nu = np.array([0., 1., 3., 3.]), np.array([1., 1., 1., .25]), 5.
    expected = np.random.default_rng(0).gamma(np.array([3., 3., 3., 2.625]), 2 / np.array([5., 6., 14., 7.25]))
    np.testing.assert_array_equal(t_precision_weights(z, omega, nu, np.random.default_rng(0)), expected)
    draws = t_precision_weights(np.repeat(z[None], 200_000, axis=0), omega, nu, np.random.default_rng(1))
    # outliers get λ < 1; a down-weighted day (ω<1) shrinks λ less
    np.testing.assert_allclose(draws.mean(axis=0), (nu + omega) / (nu + omega * z * z), rtol=.01)


@pytest.mark.parametrize("noise_state", ["op", "kind", "slot"])
def test_t_innovation_fits_with_every_noise_state_and_paths_are_finite(noise_state: str) -> None:
    panel = features.load_panel()
    train = features.role_idx(panel, "f1", "train")
    C, _ = thresholds(panel, train)
    config = ConditionalConfig(n_iter=30, burn=15, noise_state=noise_state, innovation="t")
    state = fit_conditional(panel, train, C, config)
    gaussian = fit_conditional(panel, train, C, replace(config, innovation="gaussian"))
    assert state.nu == 5. and gaussian.nu is None
    assert np.isfinite(state.sigma_eta).all() and not np.array_equal(state.sigma_eta, gaussian.sigma_eta)
    assert conditional_model(config)["name"].endswith(("" if noise_state == "op" else "_noise" + noise_state)
                                                      + "_tinnov")
    d = int(features.role_idx(panel, "f1", "val")[0])
    paths = raw_paths(state, panel, d)
    assert paths.shape == (15, 96) and np.isfinite(paths).all()
    assert not np.array_equal(paths, raw_paths(replace(state, nu=None), panel, d))


def test_innovation_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        ConditionalConfig(innovation="laplace")
    with pytest.raises(ValueError):
        ConditionalConfig(innovation="t", nu=2.)
