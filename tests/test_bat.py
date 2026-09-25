from functools import cache

import numpy as np
import torch

from gmst import bat
from gmst import evaluate as ev
from gmst import features as ft


def test_sparsemax_is_a_sparse_simplex_projection() -> None:
    z = np.random.default_rng(0).normal(size=(50, 24)) * 3
    a = bat.sparsemax(z)
    assert (a >= 0).all() and np.allclose(a.sum(1), 1) and (a == 0).any()
    np.testing.assert_allclose(bat.sparsemax(np.array([[5.0, 0.0, 0.0]])), [[1.0, 0.0, 0.0]])


def test_attention_is_local() -> None:
    alpha = bat.attention(bat.PRIOR_MU)
    assert alpha.shape == (96, 24) and np.allclose(alpha.sum(1), 1)
    assert (alpha.argmax(1) == np.arange(96) // 4).all()  # c=0: each slot attends mostly to its own hour


@cache
def fitted() -> tuple[dict, bat.BATState, int]:
    panel = ft.load_panel()
    tr = ft.role_idx(panel, "f1", "train")
    C, _ = ev.thresholds(panel, tr)
    val = ft.role_idx(panel, "f1", "val")
    return panel, bat.fit_bat(panel, tr, C, n_iter=200, burn=100), int(val[panel["op"][val] == 1][0])


def test_prediction_contract_and_determinism() -> None:
    panel, state, d = fitted()
    pred = bat.predict_bat(state, panel, d)
    assert pred["q"] is not None and pred["q"].shape == (19, 96) and np.isfinite(pred["q"]).all()
    assert (np.diff(pred["q"], axis=0) >= 0).all() and pred["risk_raw"].shape == (3,)
    np.testing.assert_array_equal(pred["paths"], bat.predict_bat(state, panel, d)["paths"])


def test_curve_fn_matches_posterior_mean_and_transfer() -> None:
    panel, state, d = fitted()
    mean, T = bat.transfer_draws(state, panel, d)
    assert (T >= 0).all()  # 생산이 늘면 전력이 줄지 않는다 (w > 0, α ≥ 0)
    q = torch.tensor(np.nan_to_num(panel["X"]["생산량"][d, ::4]), requires_grad=True)
    f = bat.curve_fn(state, panel, d)
    np.testing.assert_allclose(f(q).detach().numpy(), mean.mean(0), rtol=1e-8, atol=1e-8)
    jac = torch.autograd.functional.jacobian(f, q.detach()).numpy()
    np.testing.assert_allclose(jac, T.mean(0), rtol=1e-6, atol=1e-8)


def test_groups_from_plan() -> None:
    prod = np.zeros((4, 24))
    prod[1, :5] = 1.0  # 5 hours → type1
    prod[2, :15] = 2.0  # 15 hours → type2
    prod[3] = 3.0  # 24 hours → type3, all on
    prod[3, 23] = np.nan  # NaN → off, 23 hours still type3
    group, parent = bat.groups_from_plan(prod)
    assert group.shape == (4, 96)
    np.testing.assert_array_equal(parent, [0, 1, 1, 2, 2, 3, 3])
    assert (group[0] == 0).all()
    assert (group[1, :20] == 2).all() and (group[1, 20:] == 1).all()
    assert (group[2, :60] == 4).all() and (group[2, 60:] == 3).all()
    assert (group[3, :92] == 6).all() and (group[3, 92:] == 5).all()
    assert bat.bat_model()["name"] == "BAT"
