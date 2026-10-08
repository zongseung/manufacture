import numpy as np

from gmst import features
from gmst import plan_sensitivity as sensitivity


def test_volume_error_only_changes_issued_day_and_preserves_zero_plan() -> None:
    # Given a real panel and a deterministic volume-error scenario.
    panel = features.load_panel()
    d = int(features.role_idx(panel, "f1", "val")[0])
    original = panel["X"]["생산량"].copy()
    scenario = sensitivity.Scenario("volume_20pct", sigma=0.2)
    # When issuing an imperfect plan for this day.
    changed = sensitivity.issued_panel(panel, d, scenario, repeat=3)
    # Then historical production and all observed targets stay unchanged.
    np.testing.assert_array_equal(panel["X"]["생산량"], original)
    np.testing.assert_array_equal(changed["X"]["생산량"][:d], original[:d])
    np.testing.assert_array_equal(changed["X"]["생산량"][d + 1:], original[d + 1:])
    np.testing.assert_array_equal(changed["Y"], panel["Y"])
    np.testing.assert_array_equal(changed["X"]["생산량"][d] > 0, original[d] > 0)
    assert not np.array_equal(changed["X"]["생산량"][d], original[d])
    np.testing.assert_array_equal(changed["X"]["생산량"], sensitivity.issued_panel(panel, d, scenario, 3)["X"]["생산량"])


def test_timing_error_conserves_total_and_never_wraps_across_midnight() -> None:
    # Given a plan with production at both day boundaries.
    q = np.zeros(24)
    q[0], q[12], q[23] = 2, 4, 8
    # When the plan is one hour late, boundary production stays at 23:00.
    moved = sensitivity.perturb_plan(q, sensitivity.Scenario("late", shift=1), np.random.default_rng(0))
    # Then no end-of-day production reappears at midnight.
    assert moved.sum() == q.sum()
    assert moved[0] == 0 and moved[1] == 2 and moved[13] == 4 and moved[23] == 8


def test_zero_error_is_identity() -> None:
    # Given a plan including a missing hour.
    q = np.arange(24, dtype=float)
    q[7] = np.nan
    # When applying the clean control.
    result = sensitivity.perturb_plan(q, sensitivity.Scenario("clean"), np.random.default_rng(0))
    # Then the original values including missingness are preserved.
    np.testing.assert_array_equal(result, q)


def test_summary_averages_repeats_without_counting_extra_days() -> None:
    import polars as pl

    rows = pl.DataFrame([
        {"model": "model", "fold": "f1", "date": "2021-07-07", "scenario": scenario,
         "operating": True, "n_slots": 96, "usable_peak": True, "ae_sum": error * 96,
         "se_sum": error ** 2 * 96, "crps_sum": error * 96, "peak_ae": error,
         "plan_l1": 0.0, "changed_on_hours": 0}
        for scenario, error in (("clean", 1.0), ("volume_20pct", 2.0), ("volume_20pct", 4.0))
    ])
    summary = sensitivity.sensitivity_summary(rows)
    row = summary.filter((pl.col("fold") == "pooled") & (pl.col("stratum") == "all")
                         & (pl.col("scenario") == "volume_20pct")).row(0, named=True)
    assert row["n_days"] == 1 and row["n_slots"] == 96 and row["n_repeats"] == 2
    assert row["mae"] == 3 and row["delta_mae"] == 2
