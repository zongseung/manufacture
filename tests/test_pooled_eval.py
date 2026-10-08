import polars as pl

from gmst import pooled_eval as pe


def test_summary_pools_periods_and_paired_diff_is_c_bat_minus_other() -> None:
    rows = []
    for model, err, peak in (("C-BAT", 1., 5.), ("BAT", 2., 1.), ("M2", 3., 7.), ("B0", 4., 9.)):
        for date, period in (("d1", "dev"), ("d2", "sep")):
            rows.append({"model": model, "date": date, "period": period, "abs_sum": err * 10, "n_obs": 10,
                         "usable_peak": True, "M_true": 100., "M_hat_median": 100. + peak, "event_C90": 0.})
    days = pl.DataFrame(rows)
    s = pe.summarize(days).filter((pl.col("model") == "M2") & (pl.col("period") == "all")).row(0, named=True)
    assert s["slot_mae"] == 3. and s["peak_mae"] == 7. and s["days"] == 2
    b = {(r["comparison"], r["metric"]): r["diff"] for r in pe.paired(days, b=50).iter_rows(named=True)}
    assert b[("C-BAT − M2", "slot_mae")] == -2. and b[("C-BAT − BAT", "peak_mae")] == 4.
