import hashlib
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from gmst import DATA


def test_precipitation_resets_and_missing_hours() -> None:
    from gmst import preprocess as pp

    got = pp.precip_increment(np.array([np.nan, 0.3, np.nan, 0.5, 0.1, 0.1]))
    np.testing.assert_allclose(got, [0, 0.3, 0, 0.2, 0.1, 0])


def test_repaired_grid_and_preseptember_measurements() -> None:
    from gmst import preprocess as pp

    df = pp.build_15min()
    assert df.columns == pp.COLS
    assert df.height == 24672
    assert df["datetime"].dtype == pl.Datetime
    assert df["datetime"][0] == datetime(2021, 1, 1)
    assert df["datetime"][-1] == datetime(2021, 9, 14, 23, 45)
    assert df["datetime"].diff().drop_nulls().unique().to_list() == [timedelta(minutes=15)]
    dev = df.filter(pl.col("datetime") < datetime(2021, 9, 1))
    assert dev["전력"].null_count() == 72
    assert dev["is_missing"].sum() == 72
    assert dev["풍속"].null_count() == 0
    assert dev["강수량_증분"].null_count() == 0
    assert dev["강수량_증분"].min() >= 0
    jan24 = dev.filter(pl.col("datetime").dt.date() == date(2021, 1, 24))
    assert jan24["강수량_증분"][:4].to_list() == [0.0] * 4
    raw = pp.read_raw()
    june = raw.filter(pl.col("날짜") == 20210601)["풍속"].to_numpy()
    got = dev.filter(pl.col("datetime").dt.date() == date(2021, 6, 1))["풍속"].to_numpy()
    np.testing.assert_allclose(got[4:12], np.repeat(np.linspace(june[0], june[3], 4)[1:3], 4))


def test_audit_and_temporary_output_preserve_raw(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from gmst import preprocess as pp

    source = DATA / "okm_augumented_2021.csv"
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(pp, "OUT", tmp_path / "repaired.csv")
    pp.run(tmp_path)
    audit = json.loads((tmp_path / "data_audit.json").read_text())
    assert (audit["n_rows_raw"], audit["n_rows_15min"], audit["n_days"]) == (6168, 24672, 257)
    assert audit["zero_points"] == 74
    assert audit["zero_by_day"] == {"2021-08-28": 26, "2021-08-29": 46, "2021-09-08": 2}
    assert (audit["wind_null_hours"], audit["precip_null_hours"], audit["time_col_mismatch_rows"]) == (3, 1, 48)
    assert audit["time_col_mismatch_dates"] == ["2021-07-13", "2021-07-15"]
    assert audit["공장인원_n"] == 6151 and audit["공장인원_max_abs_err"] < 1e-6
    assert audit["평균_match"] == 6168
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    header, first, *_ = pp.OUT.read_text().splitlines()
    assert header == ",".join(pp.COLS)
    assert first.startswith("2021.01.01 00:00:00,")
