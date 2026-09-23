"""Numerical checks for the switching Gaussian AR model."""
import itertools
import math

import numpy as np
import pytest
import torch

from gmst import hmm


def test_forward_and_smoothing_match_enumerated_state_paths():
    # Given a small observed switching AR sequence.
    model = hmm.CondHMM(K=2, dz=1, seed=3).double()
    y = torch.tensor([[1.3, 0.2, 2.1]], dtype=torch.float64)
    obs = torch.ones_like(y, dtype=torch.bool)
    m = torch.tensor([[0.5, -0.2, 1.0]], dtype=torch.float64)
    op = torch.ones_like(y, dtype=torch.long)
    z = torch.zeros((1, 3, 1), dtype=torch.float64)
    a = model.trans(z)[0].detach().numpy()
    mu, sigma = model.emission(m, op)
    mu, sigma, phi = mu.detach().numpy()[0], sigma.detach().numpy()[0], model.phi().detach().numpy()
    prior = hmm.stationary(model.trans(z)[0, 0]).detach().numpy()
    weights, paths = [], list(itertools.product(range(2), repeat=3))
    for path in paths:
        weight = prior[path[0]]
        for t, k in enumerate(path):
            mean = mu[t, k]
            sd = sigma[t, k] / np.sqrt(1 - phi[k] ** 2)
            if t:
                weight *= a[t, path[t - 1], k]
                mean += phi[k] * (float(y[0, t - 1]) - mu[t - 1, path[t - 1]])
                sd = sigma[t, k]
            weight *= math.exp(-0.5 * ((float(y[0, t]) - mean) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))
        weights.append(weight)
    # When exact discrete recursion is evaluated for this observed sequence.
    logc, alpha = hmm.forward_logp(model, y, obs, m, op, z)
    gamma = hmm.backward(model, y, obs, m, op, z)
    # Then the likelihood and smoothing agree with path enumeration.
    assert float(logc.sum().detach()) == pytest.approx(math.log(sum(weights)), abs=1e-10)
    expected = np.array([[sum(w for p, w in zip(paths, weights) if p[t] == k) / sum(weights) for k in range(2)] for t in range(3)])
    np.testing.assert_allclose(gamma.detach().numpy()[0], expected, atol=1e-10)
    np.testing.assert_allclose(alpha.detach().numpy()[0, -1], expected[-1], atol=1e-10)


def test_missing_predecessor_reset_is_explicitly_approximate():
    # Given a one-state AR where the middle observation is missing.
    model = hmm.CondHMM(K=1, dz=1).double()
    with torch.no_grad():
        model.rho.zero_()
        model.psi.zero_()
        model.s.fill_(math.log(math.expm1(1)))
    y = torch.tensor([[4.0, float('nan'), 1.0]], dtype=torch.float64)
    obs = torch.isfinite(y)
    # When a masked-predecessor reset is used.
    logc, _ = hmm.forward_logp(model, y, obs, torch.zeros_like(y), torch.ones_like(y, dtype=torch.long), torch.zeros((1, 3, 1), dtype=torch.float64))
    # Then it matches stationary reset and differs from exact missing-AR integration.
    stationary_sd = 2 / math.sqrt(0.75)
    reset = torch.distributions.Normal(0.0, stationary_sd).log_prob(torch.tensor(1.0)).item()
    exact = torch.distributions.Normal(1.0, math.sqrt(5)).log_prob(torch.tensor(1.0)).item()
    assert float(logc[0, 2].detach()) == pytest.approx(reset, abs=1e-6)
    assert abs(float(logc[0, 2].detach()) - exact) > 0.05
    assert float(logc[0, 1].detach()) == pytest.approx(0.0, abs=1e-12)


def test_forward_has_finite_gradients_with_masked_values():
    # Given masked slots and ordinary parameters.
    model = hmm.CondHMM(K=3, dz=2).double()
    y = torch.tensor([[float('nan'), 2.0, float('nan'), 4.0]], dtype=torch.float64)
    # When differentiating the observed log probability.
    logc, _ = hmm.forward_logp(model, y, torch.isfinite(y), torch.zeros_like(y), torch.ones_like(y, dtype=torch.long), torch.zeros((1, 4, 2), dtype=torch.float64))
    (-logc.sum()).backward()
    # Then every trainable gradient is finite.
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_ordered_emissions_and_identified_transition_parameter_counts():
    # Given the default model and spline model.
    model = hmm.CondHMM(K=3)
    spline = hmm.CondHMM(K=3, dz=27, kan=True, M=12)
    # When reading parameterizations.
    delta, scale = model.delta(), model.sigma()
    # Then ordering, floor and nonredundant spline dimensions hold.
    assert torch.all(delta[1:] > delta[:-1]) and torch.all(scale >= 1)
    assert sum(p.numel() for p in model.parameters()) == 105
    assert spline.W.numel() == 162


def test_low_ar_start_keeps_default_and_exact_b3_baselines():
    ordinary = hmm.CondHMM(seed=0)
    low = hmm.CondHMM(seed=0, phi_start=0.02)
    independent = hmm.CondHMM(seed=0, ar=False)
    assert 0.7 < float(ordinary.phi().mean().detach()) < 0.8
    assert 0.015 < float(low.phi().mean().detach()) < 0.025
    assert torch.count_nonzero(independent.phi()) == 0


def test_pc_prior_uses_ar1_density_and_interpretable_tail():
    rate = hmm.pc_rate_for_tail(0.5, 0.05)
    distance = math.sqrt(-math.log(0.75))
    assert rate == pytest.approx(-math.log(0.05) / distance)
    phi = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
    negative_log_density = hmm.pc_nlog_prior(phi, rate)
    expected = -math.log(rate) + rate * distance - math.log(0.5 / (0.75 * distance))
    assert float(negative_log_density.detach()) == pytest.approx(expected)
    negative_log_density.backward()
    assert phi.grad is not None and torch.isfinite(phi.grad).all()
    assert torch.isfinite(hmm.pc_nlog_prior(torch.tensor([1e-8, 1 - 1e-8]), rate)).all()
    near_endpoints = torch.tensor([1e-8, 1 - 1e-8], dtype=torch.float64, requires_grad=True)
    hmm.pc_nlog_prior(near_endpoints, rate).backward()
    assert near_endpoints.grad is not None
    assert torch.isfinite(near_endpoints.grad).all()
    assert torch.count_nonzero(near_endpoints.grad) == 2
    for u, alpha in ((0.0, 0.05), (1.0, 0.05), (0.5, 0.0), (0.5, 1.0)):
        with pytest.raises(ValueError):
            hmm.pc_rate_for_tail(u, alpha)


def test_pc_prior_is_counted_once_per_observed_fit():
    panel = synthetic_panel()
    panel['Y'][1, :48] = np.nan
    blocks = hmm.make_blocks(panel, np.array([1, 2]), np.zeros_like(panel['Y']), 'A+')
    assert blocks is not None
    rate = hmm.pc_rate_for_tail(0.5, 0.05)
    plain = hmm.CondHMM(K=2, seed=0, phi_start=0.02)
    regularized = hmm.CondHMM(K=2, seed=0, phi_start=0.02)
    prior = float(hmm.pc_nlog_prior(regularized.phi(), rate).detach())
    count = int(blocks['obs'][:, 96:].sum())
    _, _, plain_history = hmm.fit_blocks(plain, blocks, epochs=1, lr=1e-5)
    _, _, pc_history = hmm.fit_blocks(regularized, blocks, epochs=1, lr=1e-5, pc_rate=rate)
    assert pc_history[0]['train_objective'] - plain_history[0]['train_objective'] == pytest.approx(prior / count, abs=1e-5)


def test_pc_prior_rejects_non_ar_and_io_models():
    with pytest.raises(ValueError):
        hmm.hmm_model('B3', pc_rate=1.0)
    with pytest.raises(ValueError):
        hmm.hmm_model('B4', decoder='io', pc_rate=1.0)


def synthetic_panel(n: int = 28):
    from datetime import date, timedelta

    import polars as pl

    from gmst.contracts import Panel
    dates = [date(2021, 3, 1) + timedelta(days=i) for i in range(n)]
    rng = np.random.default_rng(4)
    p: Panel = {'dates': dates, 'Y': rng.normal(10, 2, (n, 96)),
        'X': {k: np.zeros((n, 96)) for k in ('생산량', '기온', '풍속', '습도', '강수량_증분')},
        'is_missing': np.zeros((n, 96), bool), 'op': np.ones(n, np.int8),
        'hol': np.zeros(n, np.int8), 'dtype': np.zeros(n, np.int8),
        'dow': np.zeros(n, np.int8), 'month': np.full(n, 3, np.int8),
        'days': pl.DataFrame({'date': dates, 'f1': ['train'] * (n - 4) + ['gap'] + ['val'] * 3})}
    return p


def test_spline_partition_and_io_past_only_features():
    # Given a panel whose predecessor is partially masked.
    p = synthetic_panel()
    p['Y'][1, :48] = np.nan
    m = np.zeros_like(p['Y'])
    # When building the two feature designs.
    z = hmm.z_features(p, 'A+', np.array([2]), kan=True, M=12)
    u, v = hmm.io_features(p, m, np.array([2]))
    # Then spline design has no redundant op and residual summaries ignore masks.
    assert z.shape == (1, 96, 27)
    np.testing.assert_allclose(z[..., :24].sum(-1), 1)
    assert u.shape == (1, 96, 13) and v.shape == (1, 96, 3)
    assert u[0, 0, 11] == pytest.approx(np.nanmean(p['Y'][1]))
    assert u[0, 0, 12] == pytest.approx(np.nanmax(p['Y'][1]))


def test_masked_blocks_and_fixed_fit_reduce_observed_loss():
    # Given masked warm-up and observed target days.
    p = synthetic_panel()
    p['Y'][0] = np.nan
    blocks = hmm.make_blocks(p, np.array([1, 2]), np.zeros_like(p['Y']), 'A+')
    model = hmm.CondHMM(K=2)
    assert blocks is not None
    before = float(hmm.nll(model, blocks).detach())
    # When fitting three fixed optimizer steps.
    model, best, history = hmm.fit_blocks(model, blocks, epochs=3)
    # Then masked warm-up remains masked and optimization improves the objective.
    assert not blocks['obs'][0, :96].any()
    assert best == 3 and len(history) == 3
    assert float(hmm.nll(model, blocks).detach()) < before


def test_mc_is_date_deterministic_and_matches_phi_zero_peak_cdf():
    # Given independent Gaussian emissions and a fixed date.
    p = synthetic_panel()
    model = hmm.CondHMM(K=2, ar=False).double()
    m, thresholds = np.zeros_like(p['Y']), np.array([37.0, 42.0, 47.0])
    # When drawing paths with the default per-date seed.
    first = hmm.forecast(model, p, 2, m, thresholds, N=16000)
    second = hmm.forecast(model, p, 2, m, thresholds, N=16000)
    # Then paths are identical and MC risks converge to the analytic result.
    np.testing.assert_array_equal(first['paths'], second['paths'])
    empirical = (first['paths'].max(1)[:, None] > thresholds).mean(0)
    np.testing.assert_allclose(empirical, first['risk_raw'], atol=0.016)


def test_io_decoder_is_ordered_and_differentiable():
    # Given past-only decoder inputs.
    p = synthetic_panel()
    m = np.zeros_like(p['Y'])
    blocks = hmm.make_blocks(p, np.array([2, 3]), m, 'A+')
    assert blocks is not None
    model = hmm.CondHMM(decoder='io')
    # When evaluating conditional emissions and their likelihood.
    mu, sig, phi = model.decode(blocks['m'], blocks['opt'], blocks['u'], blocks['v'])
    loss = hmm.nll(model, blocks)
    loss.backward()
    # Then constraints and learned conditional gradients hold.
    assert torch.all(mu[..., 1:] > mu[..., :-1]) and torch.all(sig >= 1)
    assert torch.all((phi > 0) & (phi < 1))
    assert model.a.grad is not None and torch.isfinite(model.a.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA hardware is unavailable')
def test_cuda_alias_matches_model_on_default_device():
    # Given the model on the indexed default CUDA device.
    p = synthetic_panel()
    model = hmm.CondHMM().to('cuda')
    # When callers use the ordinary unindexed CUDA spelling.
    pred = hmm.forecast(model, p, 2, np.zeros_like(p['Y']), np.array([20., 30., 40.]), N=10, device='cuda')
    # Then the valid same-device forecast produces finite paths.
    assert np.isfinite(pred['paths']).all()


def test_early_stopping_restores_best_tuning_epoch():
    # Given training and tuning distributions whose preferred means disagree.
    p = synthetic_panel()
    p['Y'][1] = 50
    p['Y'][2] = -20
    m = np.zeros_like(p['Y'])
    train = hmm.make_blocks(p, np.array([1]), m, 'A+')
    tune = hmm.make_blocks(p, np.array([2]), m, 'A+')
    model = hmm.CondHMM(K=1, ar=False)
    assert train is not None and tune is not None
    # When tuning stops after deterioration.
    model, best, history = hmm.fit_blocks(model, train, tune, max_epochs=8, patience=2, lr=.5)
    # Then parameters are restored to the actual best scored epoch.
    assert best < len(history) < 8
    assert float(hmm.nll(model, tune).detach()) == pytest.approx(min(row['val_nll'] for row in history))


def test_optimizer_rejects_zero_epoch_budget():
    # Given valid observed blocks but an impossible training budget.
    p = synthetic_panel()
    blocks = hmm.make_blocks(p, np.array([1]), np.zeros_like(p['Y']), 'A+')
    assert blocks is not None
    # When zero epochs are requested.
    with pytest.raises(ValueError):
        hmm.fit_blocks(hmm.CondHMM(), blocks, blocks, max_epochs=0)
    # Then no misleading trained state can be returned.
