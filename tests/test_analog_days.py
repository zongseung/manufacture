from datetime import date

import polars as pl

from gmst import analog_days as ad


def test_pairs_match_kind_dtype_volume_and_orient_by_later_start() -> None:
    # Given three comparable days and one outside the ±10% production window.
    days = pl.DataFrame({
        "date": [date(2021, 3, d) for d in (2, 3, 4, 5)], "kind": [2, 2, 2, 2], "dtype": [0, 0, 0, 0],
        "total": [100.0, 105.0, 100.0, 150.0], "start": [9, 8, 8, 7], "start_share": [0.2, 0.3, 0.1, 0.2],
        "peak_band": [120.0, 100.0, 90.0, 50.0], "peak_all": [130.0, 110.0, 95.0, 60.0], "peak_hour": [10, 8, 8, 7],
    })
    start = ad.pairs(days, "start")
    share = ad.pairs(days, "share")
    # Then only 2↔3 and 2↔4 pair on start, oriented later − earlier; share pairs keep the same start.
    assert start.height == 2 and (start["start_b"] > start["start_a"]).all()
    assert sorted(start["d_peak_band"].to_list()) == [20.0, 30.0]
    assert share.height == 1 and share["date_b"][0] == date(2021, 3, 3) and share["d_peak_band"][0] == 10.0
    res = ad.summarize(start, 1.0)[0]
    assert res["n_pairs"] == 2 and res["mean"] == 25.0 and res["per_unit"] == 25.0


def test_day_table_excludes_sealed_and_masked_days() -> None:
    from gmst import features
    panel = features.load_panel()
    days = ad.day_table(panel)
    copies = {d for d, c in zip(panel["days"]["date"], panel["days"]["is_copy"], strict=True) if c}
    assert days["date"].max() < ad.CUTOFF and not copies & set(days["date"])
    assert (days["dtype"] < 2).all()
