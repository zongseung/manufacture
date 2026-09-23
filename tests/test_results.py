"""Validate actual saved runs with KAMP_RESULTS=/path pytest tests/test_results.py."""

import json
import os
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from gmst.run_outputs import MODEL_ID, PRED_COLS


@pytest.fixture(scope="module")
def results() -> Path:
    value = os.environ.get("KAMP_RESULTS")
    if value is None:
        pytest.skip("set KAMP_RESULTS to a completed development or smoke output directory")
    return Path(value)


def test_artifacts_preserve_variant_calibration_and_optional_status(results: Path) -> None:
    # Given an actual run, when reading artifacts, then all reported groups have p*.
    selection = json.loads((results / "selection.json").read_text())
    days = pl.read_csv(results / "oof_days.csv", infer_schema_length=None)
    assert selection["submitted"] == MODEL_ID[selection["submitted_variant"]]
    for model, variant, fold in days.select("model", "variant", "fold").unique().iter_rows():
        assert set(selection["p_star"][model][variant][fold]) == {"C50", "C75", "C90"}
    assert set(days["isotonic_status"]) == {"not_run"}
    assert selection["optional"] == {"K4": "not_run", "isotonic": "not_run"}
    assert "K4" not in days["variant"].unique().to_list()
    assert selection["n_candidates_run"] == len(selection["tried"])
    assert all(row["mae"] is None for row in selection["candidates"] if row["status"] != "run")
    for file in ("error_by_condition.csv", "fn_fp_summary.csv"):
        frame = pl.read_csv(results / file)
        assert frame.filter((pl.col("model") == "B4") & (pl.col("variant") == selection["submitted_variant"])).height


def test_issued_files_use_smoke_dates_or_are_absent(results: Path) -> None:
    # Given saved outputs, when reading issued files, then development never impersonates final.
    selection = json.loads((results / "selection.json").read_text())
    if selection["mode"] == "development" and not (results / "final_started.json").exists():
        assert not (results / "test_predictions.csv").exists()
        return
    frame = pl.read_csv(results / "test_predictions.csv")
    mask = pl.read_csv(results / "eval_mask.csv")
    assert frame.columns == PRED_COLS and frame.height == mask.height == 1344
    assert frame["model"].unique().to_list() == [selection["submitted"]]
    assert frame["track"].unique().to_list() == ["MAIN"]
    if selection["mode"] == "smoke":
        assert frame["datetime"][0] == "2021.08.18 00:00:00"
        assert frame["datetime"][-1] == "2021.08.31 23:45:00"
        assert not (results / "final_started.json").exists()
    values = frame.select("risk_C50", "risk_C75", "risk_C90").to_numpy()
    assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
    assert (np.diff(values, axis=1) <= 0).all()
    assert (np.diff(frame.select([f"q{q:02d}" for q in range(5, 100, 5)]).to_numpy(), axis=1) >= 0).all()
    assert mask.columns == ["datetime", "is_missing"]
