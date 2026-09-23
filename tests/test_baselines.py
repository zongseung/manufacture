import numpy as np
from test_backbone import synthetic_panel

from gmst import baselines as bl


def test_b0p_uses_only_complete_causal_days_and_fallback() -> None:
    # Given a recent matching day with a missing value.
    p = synthetic_panel(40)
    p["Y"][32, 0] = np.nan
    # When searching the seven-day matching window.
    ref, fallback = bl.b0p_ref(p, 39, limit=7)
    # Then the last fully observed matching day is selected.
    assert ref == 38 and not fallback
    p["dtype"][:39] = 2
    assert bl.b0p_ref(p, 39, limit=7) == (38, True)


def test_b0_fills_only_missing_lag7_slots() -> None:
    # Given a partially observed reference day.
    p = synthetic_panel()
    p["Y"][28, 4] = np.nan
    model = bl.b0_model()
    state = model["fit"](p, "f1", np.array([100., 110., 120.]))
    # When issuing the lag-seven forecast.
    pred = model["predict"](state, p, 35)
    # Then valid lag slots survive and only the absent slot uses B0p.
    ref, _ = bl.b0p_ref(p, 35)
    assert ref is not None
    assert pred["y_median"][4] == p["Y"][ref, 4]
    assert pred["y_median"][5] == p["Y"][28, 5]
    assert model["inner_state"](state) is state


def test_feature_rows_keep_missing_lags_and_protocol_boundaries() -> None:
    from gmst.lgbm_features import day_rows, lgbm_rows

    p = synthetic_panel()
    p["Y"][24] = np.nan
    days = np.array([25])
    m = np.zeros_like(p["Y"])
    original, _, names = lgbm_rows(p, days, "A+", m)
    p["X"]["기온"][25] += 900
    p["X"]["생산량"][25] += 500
    actual, _, _ = lgbm_rows(p, days, "A+", m)
    np.testing.assert_array_equal(actual, original)
    assert np.isnan(actual[:, names.index("y_lag1")]).all()
    assert len(names) == 30 and day_rows(p, days, "A+", m)[0].shape == (1, 13)
    oracle, _, oracle_names = lgbm_rows(p, days, "A+W*", m)
    np.testing.assert_array_equal(oracle[:, oracle_names.index("ob_temp")], p["X"]["기온"][25])


def test_b1_quantiles_daily_peaks_and_classifier_fallback() -> None:
    p = synthetic_panel(45)
    C = np.array([110., 140., 1000.])
    state = bl.fit_b1(p, np.arange(35), "A+", (10., 60), C, rounds=2)
    pred = bl.predict_b1(state, p, 35)
    q = pred["q"]
    assert q is not None and q.shape == (19, 96)
    assert (np.diff(q, axis=0) >= 0).all()
    np.testing.assert_array_equal(pred["y_median"], q[9])
    assert len(state["q"]) == len(state["Mq"]) == 19
    assert state["clf"][2] is None
    assert pred["risk_raw"][2] == 0
    np.testing.assert_array_equal(bl.predict_slot_median(state, p, np.array([35]))[0], pred["y_median"])


def test_b1_forecast_ignores_same_day_targets_and_weather() -> None:
    p = synthetic_panel(45)
    state = bl.fit_b1(p, np.arange(35), "A+", (10., 60), np.array([110., 140., 160.]), rounds=2)
    expected = bl.predict_b1(state, p, 35)
    p["Y"][35:] = 9000
    for values in p["X"].values():
        values[35:] = 8000
    actual = bl.predict_b1(state, p, 35)
    np.testing.assert_array_equal(actual["y_mean"], expected["y_mean"])
    np.testing.assert_array_equal(actual["q"], expected["q"])
    np.testing.assert_array_equal(actual["risk_raw"], expected["risk_raw"])


def test_hybrid_uses_cross_fitted_19_quantile_median_and_sealed_fit_inputs() -> None:
    p = synthetic_panel(35)
    tr = np.arange(25)
    C = np.array([100., 120., 140.])
    centre, state = bl.b1_centre_state(p, tr, 10., 60, C, rounds=2)
    in_sample = bl.b1_centre(p, tr, 10., 60, C, rounds=2, n_blocks=1)
    full = bl.fit_b1(p, tr, "A+", (10., 60), C, rounds=2)
    np.testing.assert_array_equal(in_sample[25], bl.predict_b1(full, p, 25)["y_median"])
    assert len(state["q"]) == 19
    assert np.isfinite(centre).all()
    assert not np.array_equal(centre[tr], in_sample[tr])
    p["Y"][25:] = 9000
    for values in p["X"].values():
        values[25:] = 8000
    perturbed, _ = bl.b1_centre_state(p, tr, 10., 60, C, rounds=2)
    np.testing.assert_array_equal(perturbed, centre)


def test_oracle_weather_changes_forecast_when_weather_drives_target() -> None:
    p = synthetic_panel(45)
    p["Y"] = np.random.default_rng(4).uniform(50, 150, (45, 96))
    p["X"]["기온"] = p["Y"].copy()
    state = bl.fit_b1(p, np.arange(35), "A+W*", (10., 60), np.array([120., 130., 140.]), rounds=10)
    expected = bl.predict_b1(state, p, 35)["y_mean"]
    p["X"]["기온"][35] += 100
    actual = bl.predict_b1(state, p, 35)["y_mean"]
    assert not np.array_equal(actual, expected)


def test_b1_inner_fit_ends_before_calibration_and_ignores_outer_labels() -> None:
    from gmst.features import inner_idx

    p = synthetic_panel(35)
    model = bl.b1_model(10., 60, rounds=2)
    C = np.array([100., 120., 140.])
    state = model["fit"](p, "f1", C)
    inner = model["inner_state"](state)
    assert inner is not None and inner["train_idx"].max() < inner_idx(p, "f1").min()
    p["Y"][30:] = 9999
    for values in p["X"].values():
        values[30:] = 8000
    changed = model["fit"](p, "f1", C)
    assert changed["mean"].model_to_string() == state["mean"].model_to_string()


def test_hybrid_cross_fit_centre_ignores_its_own_target() -> None:
    p = synthetic_panel(35)
    tr = np.arange(25)
    C = np.array([100., 120., 140.])
    original = bl.b1_centre(p, tr, 10., 60, C, rounds=10)
    p["Y"][10] += 1000
    perturbed = bl.b1_centre(p, tr, 10., 60, C, rounds=10)
    np.testing.assert_array_equal(perturbed[10], original[10])
