from datetime import date, timedelta

import numpy as np
import polars as pl

from gmst import analysis as an
from gmst.contracts import Panel


def panel_fixture() -> Panel:
    dates = [date(2021, 8, 18) + timedelta(days=d) for d in range(4)]
    y = np.arange(384, dtype=np.float64).reshape(4, 96) + 10
    return {"dates": dates, "Y": y, "X": {"생산량": np.ones((4, 96)), "기온": np.full((4, 96), 25.0)},
            "is_missing": np.zeros((4, 96), bool), "days": pl.DataFrame({"date": dates, "event_id": ["a", "b", "c", "d"]}),
            "op": np.array([1, 0, 1, 1], dtype=np.int8), "hol": np.zeros(4, dtype=np.int8),
            "dtype": np.zeros(4, dtype=np.int8), "dow": np.array([d.weekday() for d in dates], dtype=np.int8),
            "month": np.full(4, 8, dtype=np.int8)}


def test_condition_errors_when_selected_variant_is_h() -> None:
    # Given H and B1 forecasts; when selecting explicit pairs; then neither is replaced by main.
    panel = panel_fixture()
    slots = pl.DataFrame({"model": ["B4", "B4", "B1"], "variant": ["H", "main", "main"],
                          "fold": ["f4"] * 3, "date": ["2021.08.18"] * 3, "q": [0] * 3,
                          "y_true": [10.0] * 3, "y_median": [13.0, 99.0, 11.0]})
    out = an.error_by_condition(slots, panel, [("B4", "H"), ("B1", "main")])
    assert set(out.select("model", "variant").iter_rows()) == {("B4", "H"), ("B1", "main")}
    assert out.filter((pl.col("model") == "B4") & (pl.col("n") > 0))["mae"].to_list() == [3.0] * out.filter((pl.col("model") == "B4") & (pl.col("n") > 0)).height
    assert {"production", "operating", "daytype", "hour", "temperature", "holiday_adjacent"} <= set(out["condition"])


def test_fn_fp_when_variant_has_different_pstar() -> None:
    # Given risk .6 with H threshold .8; when classifying; then H has FN/TN while B1 has TP/FP.
    panel = panel_fixture()
    rows = [(m, v, "f4", d.strftime("%Y.%m.%d"), True, bool(i % 2 == 0), 0.6, 250.0 if i % 2 == 0 else 100.0)
            for m, v in [("B4", "H"), ("B1", "main")] for i, d in enumerate(panel["dates"])]
    days = pl.DataFrame(rows, schema=["model", "variant", "fold", "date", "usable_peak", "event_C90", "risk_platt_C90", "M_true"], orient="row")
    pstar = {("B4", "H", "f4"): {"C90": 0.8}, ("B1", "main", "f4"): {"C90": 0.5}}
    daily, summary = an.fn_fp(days, panel, pstar, models=[("B4", "H"), ("B1", "main")])
    selected = daily.filter((pl.col("variant") == "H") & (pl.col("p_star_kind") == "pstar"))
    assert set(selected["kind"]) == {"FN", "TN"}
    assert set(daily.filter((pl.col("model") == "B1") & (pl.col("p_star_kind") == "pstar"))["kind"]) == {"TP", "FP"}
    assert {"all", "FN", "FP"} <= set(summary["kind"])
    assert "variant" in summary.columns
    empty, empty_summary = an.fn_fp(days.head(0), panel, pstar, models=[("B4", "H")])
    assert empty.columns == daily.columns and empty_summary.columns == summary.columns
    assert empty.height == empty_summary.height == 0


def test_error_rows_when_input_empty_keep_schema() -> None:
    # Given no forecasts; when analyzing; then output stays a typed empty table.
    frame = pl.DataFrame(schema={"model": pl.String, "variant": pl.String, "date": pl.String,
                                 "q": pl.Int64, "y_true": pl.Float64, "y_median": pl.Float64})
    out = an.error_by_condition(frame, panel_fixture(), [("B4", "H")])
    assert out.columns == ["model", "variant", "condition", "bin", "n", "mae", "rmse"]
    assert out.height == 0


def test_condition_errors_when_slot_csv_has_datetime_only() -> None:
    # Given the evaluator CSV schema; when analyzing; then quarter-hour conditions are derived.
    slots = pl.DataFrame({"model": ["B4"], "variant": ["H"], "date": ["2021.08.18"],
                          "datetime": ["2021.08.18 10:15:00"], "y_true": [10.0], "y_median": [13.0]})
    out = an.error_by_condition(slots, panel_fixture(), [("B4", "H")])
    assert out.filter(pl.col("condition") == "hour")["bin"].to_list() == ["10"]
