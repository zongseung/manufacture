from datetime import date
from pathlib import Path

import polars as pl
import pytest

from gmst import preprocess


def test_real_day_policy_matches_fixed_audit() -> None:
    from gmst import splits

    days = splits.build_days(preprocess.build_15min())
    assert days.columns == splits.COLUMNS and days.height == 257
    assert days["date"].dtype == pl.Date
    assert days["event_id"].n_unique() == 142
    assert days.filter(pl.col("copy_kind") == "exact").height == 115
    assert days.filter(pl.col("is_copy")).height == 122
    groups = days.filter(pl.col("dup_size") > 1)
    assert groups.height == 160 and groups["event_id"].n_unique() == 45
    edited = days.filter(pl.col("copy_kind") == "edited")
    assert edited["date"].to_list() == list(splits.EDITED)
    assert edited["copy_of"].to_list() == ["2021.01.02", "2021.02.09", "2021.02.10", "2021.02.16", "2021.01.07", "2021.01.21", "2021.01.28"]
    jan24 = days.filter(pl.col("date") == date(2021, 1, 24))
    assert jan24["copy_kind"][0] == "exact" and jan24["copy_of"][0] == "2021.02.24"
    assert not days.filter(pl.col("date") == date(2021, 2, 24))["is_copy"][0]
    assert days.filter(pl.col("is_suspect"))["date"].to_list() == list(splits.SUSPECT)
    assert days.filter(pl.col("is_suspect"))["is_operating"].all()
    assert days["is_operating"].sum() == 195
    assert days.filter(pl.col("is_holiday"))["date"].to_list() == list(splits.HOLIDAYS_2021)
    assert dict(days.group_by("daytype").len().iter_rows()) == {"wk": 183, "sat": 37, "sun": 37}
    assert days.filter(pl.col("anomaly"))["date"].to_list() == [date(2021, 7, 13), date(2021, 7, 15), date(2021, 8, 28), date(2021, 8, 29), date(2021, 9, 8)]
    for index, fold in enumerate(splits.FOLDS):
        assert (days[fold] == "train").sum() == 186 + 14 * index
        assert (days[fold] == "gap").sum() == 1
        assert (days[fold] == "purged").sum() == 0
        assert days.filter(pl.col(fold).is_in(["val", "test"])).height == 14


def test_fold_gap_and_event_purge() -> None:
    from gmst import splits

    dates = [date(2021, 7, day) for day in range(1, 11)]
    events = ["E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E1", "E9"]
    got = splits.fold_roles(dates, events, date(2021, 7, 8), date(2021, 7, 9), "val")
    assert got == ["train", "purged", "train", "train", "train", "train", "gap", "val", "val", ""]


def test_day_table_writer_uses_requested_csv_format(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from gmst import splits

    monkeypatch.setattr(splits, "OUT", tmp_path / "days.csv")
    splits.run()
    lines = splits.OUT.read_text().splitlines()
    assert len(lines) == 258 and lines[0] == ",".join(splits.COLUMNS)
    assert lines[1].split(",")[:6] == ["2021.01.01", "E000", "1", "true", "edited", "2021.01.02"]
