import polars as pl

from gmst import report_figures


def test_examples_pick_median_error_day_per_stratum() -> None:
    dates = [f"2021.07.{d:02}" for d in range(1, 10)]
    slots = pl.DataFrame({"model": [report_figures.CBAT] * 9, "date": dates,
                          "y_true": [10.] * 9, "y_median": [11., 12., 19.] * 3})
    days = pl.DataFrame({"date": dates, "kind": [0] * 3 + [2] * 6, "high": [False] * 6 + [True] * 3})
    selected = report_figures.select_examples(slots, days)
    assert selected["date"].to_list() == [dates[1], dates[4], dates[7]]
    assert selected["stratum"].to_list() == ["Nonop", "Op", "High peak"]
