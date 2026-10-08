import numpy as np
import polars as pl

from gmst import peak_conditions


def test_peak_ties_count_each_day_once_without_counting_missing_slots() -> None:
    # Given two days with one unique peak and two equal maxima in different hours.
    slots = pl.DataFrame({"date": ["a"] * 4 + ["b"] * 4, "slot": [0, 1, 4, 5] * 2,
                          "y_true": [1., 9., np.nan, 2., 8., 1., 8., np.nan]})
    days = pl.DataFrame({"date": ["a", "b"], "M_true": [9., 8.],
                         "peak_slot_true": [1, 0], "kind": [0, 3], "high": [False, True]})
    # When all tied observed maxima receive equal daily mass.
    detail, hours = peak_conditions.peak_times(slots, days)
    # Then first-occurrence counts and tie-aware masses have the same day denominator.
    assert detail["n_peak_slots"].to_list() == [1, 2]
    all_hours = hours.filter(pl.col("stratum") == "all")
    assert all_hours["first_peak_days"].sum() == 2
    assert all_hours["tie_weighted_days"].sum() == 2
    np.testing.assert_allclose(all_hours.head(2)["tie_weighted_days"], [1.5, .5])
    assert hours.filter(pl.col("stratum") == "high")["tie_weighted_days"].sum() == 1


def test_start_hour_and_volume_bins_use_plan_and_operating_quartiles() -> None:
    # Given a nonoperating day, a day starting at 07h, and NaN plan hours.
    production = np.zeros((3, 24))
    production[1, 7:20] = 5.
    production[2, [0, 23]] = [np.nan, 1.]
    # Then the start hour is the first hour with q > 0, else -1.
    np.testing.assert_array_equal(peak_conditions.start_hours(production), [-1, 7, 23])
    # And volumes bin by quartile edges of the reference operating days; zero is "none".
    labels = peak_conditions.volume_bins(np.array([0., 1., 2.5, 3., 4., 9.]), np.array([1., 2., 3., 4.]))
    assert labels == ["none", "Q1", "Q2", "Q3", "Q4", "Q4"]
