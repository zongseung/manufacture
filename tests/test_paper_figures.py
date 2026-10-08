import polars as pl

from gmst import paper_figures


def test_examples_use_finite_median_daily_errors_in_each_stratum() -> None:
    # Given: three strata, uneven finite errors, and one missing slot per date.
    dates = [f"2021.07.{day:02}" for day in range(1, 10)]
    slots = pl.DataFrame({"model": ["BAT_conditional_gaussian"] * 18,
                          "date": dates * 2, "y_true": [10.] * 9 + [float("nan")] * 9,
                          "y_median": [11., 12., 19.] * 3 + [100.] * 9})
    days = pl.DataFrame({"date": dates, "op": [0] * 3 + [1] * 6,
                         "M_true": [20.] * 6 + [40.] * 3, "C90": [30.] * 9,
                         "usable_peak": [True] * 9})
    # When: selecting representative dates from finite daily median-forecast MAE.
    selected = paper_figures.select_examples(slots, days)
    # Then: each stratum contributes its middle-error date, unaffected by missing slots.
    assert selected["date"].to_list() == [dates[1], dates[4], dates[7]]
    assert selected["daily_mae"].to_list() == [2., 2., 2.]
