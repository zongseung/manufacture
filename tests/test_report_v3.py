import polars as pl
import pytest

from gmst.report_v3 import error_by_regime, peak_events


def test_error_by_regime_mae_and_delta() -> None:
    ctx = pl.DataFrame({"date": ["2021.07.01"] * 2, "q": [0, 1], "optype": ["k1", "k1"], "regime": ["k1_off", "k1_on"],
                        "tou": ["0", "0"], "hour": ["00", "00"]})
    slots = pl.DataFrame({"model": ["M2", "M2", "BAT", "BAT"], "variant": ["main"] * 4, "date": ["2021.07.01"] * 4,
                          "datetime": ["2021.07.01 00:00:00", "2021.07.01 00:15:00"] * 2,
                          "y_true": [10.0, 20.0, 10.0, 20.0], "y_median": [14.0, 20.0, 11.0, float("nan")]})
    t = error_by_regime(slots, ctx, ("M2", "BAT"))
    row = t.filter((pl.col("model") == "BAT") & (pl.col("condition") == "regime") & (pl.col("bin") == "k1_off")).row(0, named=True)
    assert row["mae"] == 1.0 and row["mae_minus_M2"] == -3.0
    tou = t.filter((pl.col("model") == "M2") & (pl.col("condition") == "tou")).row(0, named=True)
    assert tou["n"] == 2 and tou["mae"] == 2.0
    assert t.filter((pl.col("model") == "BAT") & (pl.col("bin") == "k1_on")).is_empty()  # NaN forecast dropped


def test_peak_events_uses_fold_pstar() -> None:
    days = pl.DataFrame({"model": ["BAT"] * 4, "variant": ["main"] * 4, "fold": ["f1", "f1", "f2", "f2"],
                         "usable_peak": [True] * 4, **{f"risk_platt_{c}": [0.9, 0.4, 0.6, 0.2] for c in ("C50", "C75", "C90")},
                         **{f"event_{c}": [1.0, 1.0, 1.0, 0.0] for c in ("C50", "C75", "C90")}})
    pstar = {("BAT", "main", "f1"): dict.fromkeys(("C50", "C75", "C90"), 0.5),
             ("BAT", "main", "f2"): dict.fromkeys(("C50", "C75", "C90"), 0.7)}
    t = peak_events(days, pstar, ("BAT",)).filter(pl.col("C") == "C50")
    pooled = t.filter(pl.col("fold") == "all").row(0, named=True)
    assert (pooled["tp"], pooled["fp"], pooled["fn"]) == (1, 0, 2)
    assert pooled["f1"] == pytest.approx(2 / 4) and pooled["precision"] == 1.0
    assert t.filter(pl.col("fold") == "f2").row(0, named=True)["f1"] == 0.0  # 0.6 < p*(f2)=0.7 → missed


def test_peak_events_f2_and_always_alarm_baseline() -> None:
    days = pl.DataFrame({"model": ["BAT"] * 4, "variant": ["main"] * 4, "fold": ["f1"] * 4, "usable_peak": [True] * 4,
                         **{f"risk_platt_{c}": [0.9, 0.2, 0.8, 0.1] for c in ("C50", "C75", "C90")},
                         **{f"event_{c}": [1.0, 1.0, 0.0, 0.0] for c in ("C50", "C75", "C90")}})
    pstar = {("BAT", "main", "f1"): dict.fromkeys(("C50", "C75", "C90"), 0.5)}
    t = peak_events(days, pstar, ("BAT",)).filter((pl.col("C") == "C50") & (pl.col("fold") == "all"))
    bat = t.filter(pl.col("model") == "BAT").row(0, named=True)  # tp=1, fp=1, fn=1
    assert bat["f2"] == pytest.approx(5 / (5 + 4 + 1)) and bat["base_rate"] == 0.5
    base = t.filter(pl.col("model") == "always_alarm").row(0, named=True)
    assert (base["tp"], base["fp"], base["fn"], base["precision"], base["recall"]) == (2, 2, 0, 0.5, 1.0)
    assert base["f1"] == pytest.approx(2 / 3) and base["f2"] == pytest.approx(5 / 6)  # b=.5: 2b/(1+b), 5b/(4b+1)
