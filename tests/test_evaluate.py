"""Synthetic checks for temporal evaluation; never read sealed observations."""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from gmst import evaluate as ev
from gmst.contracts import Panel


def panel_fixture(n: int = 35) -> Panel:
    dates = [date(2021, 6, 1) + timedelta(days=i) for i in range(n)]
    values = np.broadcast_to(np.arange(n, dtype=float)[:, None] + 100, (n, 96)).copy()
    return Panel(
        dates=dates, Y=values, X={}, is_missing=np.zeros((n, 96), dtype=bool),
        days=pl.DataFrame({"date": dates, "f1": ["train"] * 25 + ["val"] * (n - 25),
                           "f2": [""] * n, "f3": [""] * n, "f4": [""] * n}),
        op=np.ones(n, dtype=np.int8), hol=np.zeros(n, dtype=np.int8),
        dtype=np.zeros(n, dtype=np.int8), dow=np.zeros(n, dtype=np.int8),
        month=np.full(n, 6, dtype=np.int8),
    )


def day_fixture(fold: str, probabilities: list[float], events: list[int],
                variant: str = "main", start: date = date(2021, 6, 1)) -> pl.DataFrame:
    frame = pl.DataFrame({"model": ["B4"] * len(events), "variant": [variant] * len(events),
                          "fold": [fold] * len(events), "date": [start + timedelta(days=i) for i in range(len(events))],
                          "usable_peak": [True] * len(events)})
    return frame.with_columns(*[pl.Series(f"risk_raw_{c}", probabilities) for c in ev.CS],
                              *[pl.Series(f"event_{c}", events, dtype=pl.Float64) for c in ev.CS])


def test_point_metrics_use_finite_pairs_and_actual_quantiles() -> None:
    # Given masked observations and asymmetric predictions.
    y = np.array([1.0, np.nan, 3.0])
    yh = np.array([2.0, 100.0, 1.0])
    # When metrics consume the same finite pairs.
    measured = ev.mae(y, yh), ev.rmse(y, yh), ev.crps(np.array([0.0]), np.ones((1, 19)))
    # Then denominators and pinball integration are exact.
    assert measured == ((1.5, 2), (pytest.approx(np.sqrt(2.5)), 2), (pytest.approx(1.0), 1))
    assert ev.coverage(np.array([1.0, 5.0, np.nan]), np.zeros(3), np.array([2.0, 4.0, 1.0])) == (0.5, 2)


def test_auc_ties_and_missing_event_class_are_explicit() -> None:
    # Given one tied ranking and one event-free sample.
    p = np.array([0.1, 0.4, 0.35, 0.8])
    # When discrimination metrics are evaluated.
    ranked = ev.auc(p, np.array([0.0, 0.0, 1.0, 1.0]))
    # Then pair ordering and undefined cases are preserved.
    assert ranked == 0.75
    assert ev.auc(np.full(3, 0.5), np.array([0.0, 1.0, 1.0])) == 0.5
    assert np.isnan(ev.auc(p, np.zeros(4)))
    assert np.isnan(ev.prf(np.array([True, False]), np.zeros(2, dtype=bool))["f1"])


def test_observed_peak_excludes_hidden_high_slot() -> None:
    # Given a large forecast in an unobserved slot.
    paths = np.ones((3, 96))
    paths[:, 10], paths[:, 40] = 500, [170, 175, 190]
    observed = np.ones(96, dtype=bool)
    observed[10] = False
    # When restricting maxima to observed slots.
    maxima = ev.obs_max(paths, observed)
    # Then hidden forecast values do not affect scored maxima.
    np.testing.assert_array_equal(maxima, [170, 175, 190])
    assert np.isnan(ev.obs_max(np.ones(96), np.zeros(96, dtype=bool)))


def test_thresholds_and_bootstrap_dates_derive_from_masks() -> None:
    # Given one wholly masked day and one partially masked usable point day.
    panel = panel_fixture()
    panel["Y"][25] = np.nan
    panel["Y"][26, :10] = np.nan
    # When deriving thresholds and paired bootstrap dates.
    thresholds, n = ev.thresholds(panel, np.arange(25, dtype=np.int64))
    dates = ev.bootstrap_days(panel)
    # Then partial days remain in point comparisons; thresholds use train only.
    np.testing.assert_allclose(thresholds, [112, 118, 121.6])
    assert n == 25
    np.testing.assert_array_equal(dates, np.arange(26, 35))


def test_platt_empty_identity_and_single_class_constant() -> None:
    # Given empty calibration and a one-class sample.
    p = np.array([0.0, 0.2, 0.7, 1.0])
    # When fitting calibrated probabilities.
    empty = ev.platt_apply(ev.platt_fit(np.array([]), np.array([])), p)
    single = ev.platt_apply(ev.platt_fit(p, np.zeros(4)), p)
    # Then empty means exact identity and one class yields finite constant risk.
    np.testing.assert_array_equal(empty, p)
    assert np.isfinite(single).all() and np.ptp(single) == pytest.approx(0.0)
    assert 0 < single[0] < 0.5


def test_temporal_calibration_ignores_outer_labels_and_other_variants() -> None:
    # Given inner rows for matching fold and a conflicting other variant.
    inner = pl.concat([day_fixture("f1", [0.1, 0.3, 0.7, 0.9], [0, 0, 1, 1]),
                       day_fixture("f1", [0.1, 0.3, 0.7, 0.9], [1, 1, 0, 0], variant="H")])
    outer = day_fixture("f1", [0.2, 0.8], [0, 1], start=date(2021, 7, 1))
    changed = outer.with_columns(*[(1 - pl.col(f"event_{c}")).alias(f"event_{c}") for c in ev.CS])
    # When the outer labels change.
    calibrated, ci = ev.calibrate_oof(outer, inner=inner)
    perturbed, pi = ev.calibrate_oof(changed, inner=inner)
    # Then both calibrated outer risks and inner p* inputs remain unchanged.
    np.testing.assert_array_equal(calibrated.select("^risk_platt.*$").to_numpy(), perturbed.select("^risk_platt.*$").to_numpy())
    assert ci is not None and pi is not None and ci.equals(pi)
    assert calibrated["calibration_status_platt_C50"].to_list() == ["ok", "ok"]
    assert calibrated["n_cal_platt_C50"].to_list() == [4, 4]


def test_final_calibration_uses_only_matching_pretest_reference() -> None:
    # Given a matching OOF reference plus another variant and a future row.
    ref = pl.concat([day_fixture("f1", [0.1, 0.9], [0, 1], variant="H"),
                     day_fixture("f2", [0.1, 0.9], [1, 0]),
                     day_fixture("f3", [0.1], [1], variant="H", start=date(2021, 9, 2))])
    final = day_fixture("test", [0.2, 0.8], [0, 1], variant="H", start=date(2021, 9, 1))
    # When fitting the final calibrator.
    calibrated, _ = ev.calibrate_oof(final, ref=ref)
    # Then exactly the same variant's pretest rows fit it.
    assert calibrated["n_cal_platt_C50"].to_list() == [2, 2]
    assert calibrated["risk_platt_C50"][0] < calibrated["risk_platt_C50"][1]


def test_pstar_has_all_expected_keys_and_empty_default() -> None:
    # Given a single calibrated group and additional required groups.
    outer = day_fixture("f1", [0.2], [1], start=date(2021, 7, 1))
    _, inner = ev.calibrate_oof(outer, inner=day_fixture("f1", [0.1, 0.9], [0, 1]))
    keys = [("B4", "main", "f1"), ("BB", "main", "f2"), ("B4", "KAN", "f4")]
    # When producing thresholds.
    table = ev.p_star_table(inner, keys)
    # Then every key has every threshold, with visible fallback counts.
    assert set(table) == set(keys)
    assert table[("BB", "main", "f2")] == {c: 0.5 for c in ev.CS}
    missing = ev.p_star_status(inner, keys).filter(pl.col("model") == "BB")
    assert missing["status"].to_list() == ["default_empty"] * 3
    assert missing["n_cal"].to_list() == [0] * 3


def test_pav_pools_ties_and_inversions() -> None:
    # Given contradictory tied scores and an order inversion.
    p, events = np.array([0.1, 0.1, 0.5, 0.9]), np.array([1.0, 0.0, 0.0, 1.0])
    # When fitting isotonic calibration.
    calibrated = ev.pav_apply(ev.pav_fit(p, events), np.array([0.0, 0.1, 0.5, 0.9, 1.0]))
    # Then equal values share the pooled monotone mean.
    np.testing.assert_allclose(calibrated, [1 / 3, 1 / 3, 1 / 3, 1, 1])


def test_bootstrap_paired_differences_and_dm_zero_variance() -> None:
    # Given a known constant paired improvement.
    loss = np.full(20, -2.0)
    # When bootstrapping and computing the auxiliary normal statistic.
    interval = ev.block_bootstrap(loss, np.ones(20), B=100, seed=4)
    # Then all replicates retain the exact difference.
    assert interval == (-2.0, -2.0, -2.0)
    assert ev.dm_test(loss) == (-np.inf, 0.0)


def test_rolling_origin_naive_models_have_real_inner_predictions() -> None:
    # Given a finite synthetic fold with seven calibration days.
    from gmst import baselines as bl
    panel = panel_fixture()
    # When both naive models issue calibration and outer predictions.
    outputs = [ev.rolling_origin(model, panel, folds=("f1",)) for model in (bl.b0_model(), bl.b0p_model())]
    # Then every group has seven real inner days and 96 slots per outer day.
    for slots, days, states, inner in outputs:
        assert slots.height == 960 and days.height == 10 and inner.height == 7
        assert np.isfinite(inner["risk_raw_C50"].to_numpy()).all()
        assert list(states) == ["f1"]
        assert slots.columns == ev.SLOT_COLS and days.columns == ev.DAY_COLS


def test_risk_metrics_require_pstars_and_skip_optional_iso() -> None:
    # Given rolled, calibrated predictions and complete p* keys.
    from gmst import baselines as bl
    panel = panel_fixture()
    slots, days, _, inner = ev.rolling_origin(bl.b0p_model(), panel, folds=("f1",))
    days, inner = ev.calibrate_oof(days, inner=inner)
    pstars = ev.p_star_table(inner, [("B0p", "main", "f1")])
    # When composing point and risk metrics.
    metrics = pl.concat([ev.point_metrics(slots, days, panel), ev.risk_metrics(days, panel, pstars)])
    # Then iso is absent, schema is stable, and incomplete p* raises.
    assert metrics.columns == list(ev.METRIC_SCHEMA)
    assert "brier_mean_platt" in metrics["metric"]
    assert not any("iso" in name for name in metrics["metric"])
    with pytest.raises(KeyError, match="Missing p"):
        ev.risk_metrics(days, panel, {})


def test_climatology_and_risk_checks_preserve_variant_identity() -> None:
    # Given opposing predictions for two variants with the same model identifier.
    good = day_fixture("f1", [0.1, 0.9], [0, 1])
    bad = day_fixture("f1", [0.9, 0.1], [0, 1], variant="H")
    days = pl.concat([good, bad]).with_columns(*[pl.col(f"risk_raw_{c}").alias(f"risk_platt_{c}") for c in ev.CS],
                                              *[pl.lit(0.5).alias(f"climcond_{c}") for c in ev.CS])
    # When testing risk discrimination.
    checked = ev.risk_check(days)
    # Then each variant has its own judgment.
    assert checked["B4/main"]["discrimination_missing"] is False
    assert checked["B4/H"]["discrimination_missing"] is True


def test_climatology_empty_cell_falls_back_to_operating_group() -> None:
    # Given a single populated cell in each operating group.
    clim = ev.climatology(np.array([200.0, 100.0]), np.array([1, 0], dtype=np.int64),
                         np.array([0, 0], dtype=np.int64), np.array([150.0, 180.0, 220.0]))
    # When asking for unseen Saturday in operating class.
    probability = ev.clim_prob(clim, 1, 1)
    # Then the operating-group rate is used.
    np.testing.assert_array_equal(probability, [1.0, 1.0, 0.0])


def test_compare_has_distinct_point_and_peak_denominators() -> None:
    # Given a partial outage day and two deterministic forecasts.
    from gmst import baselines as bl
    panel = panel_fixture()
    panel["Y"][25, :10] = np.nan
    panel["is_missing"][25, :10] = True
    slots, days, _, inner = ev.rolling_origin(bl.b0p_model(), panel, folds=("f1",))
    days, _ = ev.calibrate_oof(days, inner=inner)
    worse = slots.with_columns((pl.col("y_median") - 1).alias("y_median"))
    # When comparing the two models with paired day resampling.
    comparison = ev.compare(slots, days, worse, days, panel, "B4-B1", B=50)
    # Then partially observed days contribute point losses but no peak risk.
    by_metric = {row["metric"]: row for row in comparison}
    assert by_metric["mae"]["n_days"] == 10 and by_metric["mae"]["n_valid_days"] == 10
    assert by_metric["mae"]["n_obs"] == 950 and by_metric["mae"]["delta"] == pytest.approx(-1)
    assert by_metric["brier_mean"]["n_days"] == 10 and by_metric["brier_mean"]["n_valid_days"] == 9
    assert by_metric["crps"]["n_obs"] == 0


def test_gate_requires_both_improvements_and_selection_stays_b4() -> None:
    # Given main misses Brier, H passes both, and IO has lower MAE but fails gate.
    boot = pl.DataFrame({"comparison": ["B4-B1"] * 2 + ["B4-H-B1"] * 2,
                         "metric": ["mae", "brier_mean"] * 2, "ci_hi": [-1.0, 0.1, -0.5, -0.01]})
    candidates: dict[str, ev.Candidate] = {
        "main": {"mae": 2.0, "brier_mean": 0.2, "gate_met": ev.gate_pass(boot, "B4-B1"), "gap": {}},
        "H": {"mae": 1.5, "brier_mean": 0.1, "gate_met": ev.gate_pass(boot, "B4-H-B1"), "gap": {}},
        "IO": {"mae": 1.0, "brier_mean": 0.3, "gate_met": False, "gap": {}},
    }
    # When selecting the candidate.
    selected = ev.select_variant(candidates)
    # Then passed candidates are preferred without ever selecting the B1 benchmark.
    assert selected["submitted"] == "B4-H" and selected["model"] == "B4" and selected["variant"] == "H"
    assert selected["tried"] == ["main", "H", "IO"] and selected["n_candidates_run"] == 3


def test_real_nonsealed_panel_has_53_bootstrap_days_and_51_peak_days() -> None:
    # Given the normal loader, which seals every September input.
    from gmst import features as ft
    panel = ft.load_panel()
    # When deriving eligible OOF dates.
    days = ev.bootstrap_days(panel)
    # Then mask policies determine distinct point and peak populations.
    assert len(days) == 53 and int(ft.usable_peak(panel)[days].sum()) == 51
    assert all(panel["dates"][int(i)] <= date(2021, 8, 31) for i in days)


def test_nll_is_evaluated_after_issuance_and_weighted_by_observations() -> None:
    # Given a model that records which days have already been issued.
    from gmst.contracts import FloatArray, IntArray, Model, Prediction
    panel = panel_fixture()
    issued: set[int] = set()

    def predict(state: float, p: Panel, d: int) -> Prediction:
        issued.add(d)
        return ev.point_pred(np.full(96, state), np.array([100.0, 120.0, 130.0]))

    def nll(state: float, p: Panel, idx: IntArray) -> tuple[float, int]:
        assert set(range(25, 35)) <= issued
        return float(idx[0]), int(idx[0] - 24)

    def fit(p: Panel, fold: str, c: FloatArray) -> float:
        return 120.0

    model: Model[float] = {"name": "B4", "fit": fit, "predict": predict,
                           "inner_state": lambda state: state, "evaluate_nll": nll}
    # When rolling and aggregating likelihood diagnostics.
    slots, days, _, _ = ev.rolling_origin(model, panel, folds=("f1",))
    metrics = ev.point_metrics(slots, days, panel)
    # Then NLL uses the callback's actual observation counts.
    actual = metrics.filter((pl.col("metric") == "nll") & (pl.col("fold") == "f1") & (pl.col("stratum") == "all"))
    assert actual["value"][0] == pytest.approx(float(np.average(np.arange(25, 35), weights=np.arange(1, 11))))
    assert actual["n"][0] == 55


def test_compare_rejects_different_slot_masks_even_when_counts_match() -> None:
    # Given equal point counts that refer to different observed slots.
    from gmst import baselines as bl
    panel = panel_fixture()
    slots, days, _, inner = ev.rolling_origin(bl.b0p_model(), panel, folds=("f1",))
    days, _ = ev.calibrate_oof(days, inner=inner)
    a = slots.with_row_index().with_columns(pl.when(pl.col("index") == 0).then(float("nan")).otherwise(pl.col("y_median")).alias("y_median")).drop("index")
    b = slots.with_row_index().with_columns(pl.when(pl.col("index") == 1).then(float("nan")).otherwise(pl.col("y_median")).alias("y_median")).drop("index")
    # When requesting paired inference.
    with pytest.raises(ValueError, match="masks differ"):
        ev.compare(a, days, b, days, panel, "B4-B1", B=10)
    # Then a spurious same-count comparison cannot pass.


def test_cv_calibration_rejects_outer_reference_substitution() -> None:
    # Given a CV fold and an attempted external OOF reference.
    outer = day_fixture("f1", [0.1, 0.9], [0, 1], start=date(2021, 7, 1))
    # When trying to replace its train-only inner calibration with other folds.
    with pytest.raises(ValueError, match="reference.*test"):
        ev.calibrate_oof(outer, ref=day_fixture("f2", [0.2, 0.8], [0, 1]))
    # Then the temporal contract cannot silently become leave-fold-out calibration.


def test_compare_rejects_changed_risk_events() -> None:
    # Given forecasts that use different risk-event definitions on the same dates.
    from gmst import baselines as bl
    panel = panel_fixture()
    slots, days, _, inner = ev.rolling_origin(bl.b0p_model(), panel, folds=("f1",))
    days, _ = ev.calibrate_oof(days, inner=inner)
    changed = days.with_columns((1 - pl.col("event_C90")).alias("event_C90"))
    # When attempting paired Brier inference.
    with pytest.raises(ValueError, match="risk events"):
        ev.compare(slots, days, slots, changed, panel, "B4-B1", B=10)
    # Then comparison fails despite equal denominators.


def test_compare_serial_correlation_reports_block_sensitivity() -> None:
    # Given a smoothly increasing paired loss difference over consecutive days.
    from gmst import baselines as bl
    panel = panel_fixture()
    slots, days, _, inner = ev.rolling_origin(bl.b0p_model(), panel, folds=("f1",))
    days, _ = ev.calibrate_oof(days, inner=inner)
    worse = slots.with_columns((pl.col("y_median") - pl.col("date").rank("dense")).alias("y_median"))
    # When comparing losses.
    row = ev.compare(slots, days, worse, days, panel, "B4-B1", metrics=("mae",), B=50)[0]
    # Then the sensitivity interval accompanies the day-resampling interval.
    assert row["serial_correlation"] is True and row["block7_status"] == "ok"
    assert np.isfinite(float(str(row["block7_ci_lo"]))) and np.isfinite(float(str(row["block7_ci_hi"])))


def test_rolling_scores_observed_slot_risk_without_mutating_issued_prediction() -> None:
    # Given three paths with a high peak only at an unobserved target slot.
    from gmst.contracts import FloatArray, Model, Prediction
    panel = panel_fixture()
    panel["Y"][25, 10] = np.nan
    panel["is_missing"][25, 10] = True
    paths = np.ones((3, 96))
    paths[:, 10], paths[:, 40] = 500, [105, 110, 115]
    issued: list[Prediction] = []

    def predict(state: FloatArray, p: Panel, d: int) -> Prediction:
        q = np.quantile(paths, ev.TAUS, axis=0)
        pred: Prediction = {"y_mean": paths.mean(axis=0), "y_median": q[9], "q": q, "paths": paths,
                            "M_hat_median": 500.0, "M_hat_mean": 500.0, "peak_time_mode": 10,
                            "risk_raw": np.ones(3)}
        issued.append(pred)
        return pred

    model: Model[FloatArray] = {"name": "B4", "fit": lambda p, f, c: c,
                                "predict": predict, "inner_state": lambda state: state}
    # When issuing and scoring the model.
    _, days, _, _ = ev.rolling_origin(model, panel, folds=("f1",))
    # Then scoring uses the observed set without changing the as-issued probabilities.
    assert days["M_hat_median"][0] == 110 and days["risk_raw_C90"][0] == 0
    assert issued[0]["risk_raw"][2] == 1 and issued[0]["M_hat_median"] == 500
    assert days["M_hat_median"][1] == 500


def test_empty_risk_group_remains_visible_as_failed_check() -> None:
    # Given a model with no usable peak day.
    days = day_fixture("f1", [0.1], [0]).with_columns(pl.lit(False).alias("usable_peak"),
        *[pl.col(f"risk_raw_{c}").alias(f"risk_platt_{c}") for c in ev.CS],
        *[pl.lit(0.5).alias(f"climcond_{c}") for c in ev.CS])
    # When checking the model.
    checks = ev.risk_check(days)
    # Then insufficient evidence is reported rather than dropping the model.
    assert "B4/main" in checks and checks["B4/main"]["discrimination_missing"] is True


def test_final_pstar_rows_cannot_include_test_observations() -> None:
    # Given valid pretest OOF but an invalid calibration day inside final issuance.
    final = day_fixture("test", [0.2], [1], start=date(2021, 9, 1))
    ref = day_fixture("f1", [0.1, 0.9], [0, 1])
    # When calibrating final risk and p* inputs.
    with pytest.raises(ValueError, match="precede outer"):
        ev.calibrate_oof(final, inner=final, ref=ref)
    # Then final target labels cannot choose the operating threshold.


def test_isotonic_placeholders_do_not_create_unrun_metrics() -> None:
    # Given a skipped isotonic run with placeholder columns from a CSV schema.
    from gmst import baselines as bl
    panel = panel_fixture()
    _, days, _, inner = ev.rolling_origin(bl.b0p_model(), panel, folds=("f1",))
    days, inner = ev.calibrate_oof(days, inner=inner)
    placeholders = days.with_columns(*[pl.lit(float("nan")).alias(f"risk_iso_{c}") for c in ev.CS])
    # When reporting metrics.
    metrics = ev.risk_metrics(placeholders, panel, ev.p_star_table(inner, [("B0p", "main", "f1")]))
    # Then columns alone cannot claim that optional calibration ran.
    assert not any("iso" in name for name in metrics["metric"])


def test_block_sensitivity_uses_calendar_days_across_masked_gap() -> None:
    # Given correlated errors with one completely masked calendar day.
    from gmst import baselines as bl
    panel = panel_fixture()
    panel["Y"][29] = np.nan
    slots, days, _, inner = ev.rolling_origin(bl.b0p_model(), panel, folds=("f1",))
    days, _ = ev.calibrate_oof(days, inner=inner)
    worse = slots.with_columns((pl.col("y_median") - pl.col("date").rank("dense")).alias("y_median"))
    numerator = -96.0 * np.arange(1.0, 11.0)
    denominator = np.full(10, 96.0)
    numerator[4], denominator[4] = 0, 0
    expected = ev.block_bootstrap(numerator, denominator, B=80, seed=0, block_length=7)
    # When computing the moving-block sensitivity interval.
    row = ev.compare(slots, days, worse, days, panel, "B4-B1", metrics=("mae",), B=80)[0]
    # Then a seven-day block includes the zero-weight missing calendar day.
    assert row["block7_ci_lo"] == pytest.approx(expected[1])
    assert row["block7_ci_hi"] == pytest.approx(expected[2])


@pytest.mark.parametrize("has_daily_model", [False, True])
def test_masked_point_risk_and_separate_day_classifier_policy(has_daily_model: bool) -> None:
    # Given a peak forecast entirely inside a missing observation slot.
    from gmst.contracts import Model, Prediction
    panel = panel_fixture()
    panel["Y"][25, 10] = np.nan
    panel["is_missing"][25, 10] = True
    y = np.full(96, 100.0)
    y[10] = 500
    pred = ev.point_pred(y, np.array([112.0, 118.0, 121.6]))
    if has_daily_model:
        pred["q"] = np.tile(y, (19, 1))
    model: Model[Prediction] = {"name": "B1" if has_daily_model else "B0", "fit": lambda p, f, c: pred,
                                "predict": lambda state, p, d: state, "inner_state": lambda state: state}
    # When scoring evaluation-day risk.
    _, days, _, _ = ev.rolling_origin(model, panel, folds=("f1",))
    # Then point forecasts can be masked; a separate daily classifier remains as issued.
    assert days["risk_raw_C90"][0] == float(has_daily_model)
    assert pred["risk_raw"][2] == 1 and pred["M_hat_median"] == 500


def test_completely_masked_day_contributes_no_risk_score() -> None:
    # Given a whole target day masked by the panel policy.
    from gmst import baselines as bl
    panel = panel_fixture()
    panel["Y"][25] = np.nan
    slots, days, _, _ = ev.rolling_origin(bl.b0p_model(), panel, folds=("f1",))
    # When evaluating daily risk.
    measured = ev.point_metrics(slots, days, panel)
    # Then predictions may exist but the unknown event has no score weight.
    assert not days["usable_peak"][0] and np.isnan(days["event_C90"][0])
    row = measured.filter((pl.col("fold") == "f1") & (pl.col("stratum") == "all") & (pl.col("metric") == "brier_raw_C90"))
    assert row["n"][0] == 9


def test_new_candidate_after_refresh_gets_explicit_optional_status() -> None:
    # Given calibrated rows concatenated with a newly attempted raw candidate.
    outer = day_fixture("f1", [0.2, 0.8], [0, 1], start=date(2021, 7, 1))
    old_inner = day_fixture("f1", [0.1, 0.9], [0, 1])
    calibrated, inner = ev.calibrate_oof(outer, inner=old_inner)
    assert inner is not None
    all_days = pl.concat([calibrated, outer.with_columns(pl.lit("H").alias("variant"))], how="diagonal_relaxed")
    all_inner = pl.concat([inner, old_inner.with_columns(pl.lit("H").alias("variant"))], how="diagonal_relaxed")
    # When recalibrating after the candidate is appended.
    refreshed, refreshed_inner = ev.calibrate_oof(all_days, inner=all_inner)
    # Then nulls introduced by the schema union become explicit not-run status.
    assert refreshed["isotonic_status"].to_list() == ["not_run"] * 4
    assert refreshed_inner is not None and refreshed_inner["isotonic_status"].to_list() == ["not_run"] * 4
