"""Repair measurements without applying modeling or evaluation policy."""
import json
from datetime import datetime
from pathlib import Path
from typing import Final, TypedDict

import numpy as np
import polars as pl

from gmst import DATA, RESULTS
from gmst.contracts import FloatArray

RAW: Final = DATA / "okm_augumented_2021.csv"
OUT: Final = DATA / "okm_15min_2021.csv"
POWER: Final = ["15분", "30분", "45분", "60분"]
COLS: Final = ["datetime", "전력", "생산량", "기온", "풍속", "습도", "강수량_증분", "전기요금(계절)", "인건비", "is_missing"]


class Audit(TypedDict):
    n_rows_raw: int
    n_rows_15min: int
    n_days: int
    zero_points: int
    zero_by_day: dict[str, int]
    wind_null_hours: int
    precip_null_hours: int
    time_col_mismatch_rows: int
    time_col_mismatch_dates: list[str]
    공장인원_max_abs_err: float
    공장인원_n: int
    평균_match: int


def read_raw() -> pl.DataFrame:
    """Use physical row order because two recorded hour columns are corrupted."""
    return pl.read_csv(RAW, infer_schema_length=None).with_columns(
        pl.int_range(pl.len(), dtype=pl.Int64).over("날짜").alias("hour")
    )


def precip_increment(cumulative: FloatArray) -> FloatArray:
    """Difference continuous cumulative rainfall, retaining the last known reading."""
    increments = np.zeros_like(cumulative, dtype=np.float64)
    previous = np.nan
    for index, current in enumerate(cumulative):
        if np.isnan(current):
            continue
        delta = current - previous
        increments[index] = current if np.isnan(previous) or delta < 0 else delta
        previous = current
    return increments


def build_15min() -> pl.DataFrame:
    """Return interval-start measurements; only physical outages become null."""
    raw = read_raw()
    wind = raw["풍속"].to_numpy().astype(np.float64)
    known = np.flatnonzero(np.isfinite(wind))
    wind = np.interp(np.arange(raw.height), known, wind[known])
    hourly = raw.with_columns(
        pl.Series("풍속", wind),
        pl.Series("강수량_증분", precip_increment(raw["강수량"].to_numpy().astype(np.float64))),
        pl.col("날짜").cast(pl.String).str.strptime(pl.Date, "%Y%m%d").alias("date"),
    )
    slots = hourly.unpivot(on=POWER, index=["date", "hour", *COLS[2:-1]], variable_name="slot", value_name="전력")
    return slots.with_columns(
        (pl.col("date").cast(pl.Datetime) + pl.duration(hours=pl.col("hour"))
         + pl.duration(minutes=pl.col("slot").replace_strict(dict(zip(POWER, [0, 15, 30, 45])), return_dtype=pl.Int64))).alias("datetime"),
        (pl.col("전력") == 0).alias("is_missing"),
        pl.when(pl.col("전력") == 0).then(None).otherwise(pl.col("전력")).cast(pl.Float64).alias("전력"),
        pl.col(COLS[2:-1]).cast(pl.Float64),
    ).select(COLS).sort("datetime")


def audit() -> Audit:
    """Return fixed data-integrity metadata, without forecasting statistics."""
    raw = read_raw()
    power = raw.select(POWER).to_numpy().astype(np.float64)
    sums = power.sum(axis=1)
    staff = raw["공장인원"].to_numpy().astype(np.float64)
    production = raw["생산량"].to_numpy().astype(np.float64)
    valid = (sums > 0) & np.isfinite(staff)
    mismatch = raw.filter(pl.col("시간") != pl.col("hour"))
    zeros = raw.select("날짜").with_columns(pl.Series("count", (power == 0).sum(axis=1))).group_by("날짜").agg(pl.col("count").sum()).filter(pl.col("count") > 0).sort("날짜")
    return {
        "n_rows_raw": raw.height, "n_rows_15min": raw.height * 4,
        "n_days": raw["날짜"].n_unique(), "zero_points": int((power == 0).sum()),
        "zero_by_day": {datetime.strptime(str(day), "%Y%m%d").date().isoformat(): int(count) for day, count in zeros.iter_rows()},
        "wind_null_hours": raw["풍속"].null_count(), "precip_null_hours": raw["강수량"].null_count(),
        "time_col_mismatch_rows": mismatch.height,
        "time_col_mismatch_dates": [datetime.strptime(str(day), "%Y%m%d").date().isoformat() for day in sorted(mismatch["날짜"].unique().to_list())],
        "공장인원_max_abs_err": float(np.abs(staff[valid] - production[valid] / sums[valid]).max()),
        "공장인원_n": int(valid.sum()),
        "평균_match": int((raw["평균"].to_numpy() == np.floor(power.mean(axis=1) + 0.5)).sum()),
    }


def run(out: Path = RESULTS) -> None:
    out.mkdir(parents=True, exist_ok=True)
    build_15min().write_csv(OUT, datetime_format="%Y.%m.%d %H:%M:%S")
    (out / "data_audit.json").write_text(json.dumps(audit(), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run()
