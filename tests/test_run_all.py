"""CLI isolation, packaging and the issued submission schema."""

import subprocess
import sys
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from gmst import ROOT
from gmst.contracts import Prediction


def test_cli_rejects_smoke_final_before_loading_data() -> None:
    # Given incompatible execution modes, when invoked, then no result is emitted.
    result = subprocess.run(
        [sys.executable, "-m", "gmst.run_all", "--smoke", "--final"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "not allowed" in result.stderr


def test_final_requires_frozen_selection(tmp_path: Path) -> None:
    # Given no selection artifact, when final is requested, then fail before unseal.
    result = subprocess.run(
        [sys.executable, "-m", "gmst.run_all", "--final", "--out", str(tmp_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "selection.json" in result.stderr
    assert not list(tmp_path.glob("test_*"))


def test_package_is_allowlisted(tmp_path: Path) -> None:
    from gmst.run_all import build_package

    # Given the workspace, when packaged, then only delivery roots are included.
    archive = build_package(tmp_path / "source.zip")
    with zipfile.ZipFile(archive) as package:
        names = package.namelist()
    assert {"README.md", "requirements.txt", "uv.lock", "gmst/run_all.py"} <= set(names)
    assert all(
        Path(name).parts[0]
        in {
            "gmst",
            "tests",
            "notebooks",
            "results",
            "5. 자원 최적화 AI 데이터셋",
            "README.md",
            "requirements.txt",
            "pyproject.toml",
            "uv.lock",
            "PONYTAIL-DEBT.md",
        }
        for name in names
    )
    assert not any(part.startswith(".") or part == "__pycache__" for name in names for part in Path(name).parts)


def test_issued_submission_preserves_selected_variant_and_full_day() -> None:
    from gmst.run_outputs import PRED_COLS, submission_rows

    # Given issued paths and a calibrated H risk, when formatted, then 96 slots repeat full-day values.
    prediction: Prediction = {
        "y_mean": np.full(96, 42.0),
        "y_median": np.full(96, 40.0),
        "q": np.repeat(np.arange(19.0)[:, None], 96, axis=1),
        "paths": None,
        "M_hat_median": 45.0,
        "M_hat_mean": 46.0,
        "peak_time_mode": 45,
        "risk_raw": np.array([0.9, 0.8, 0.7]),
    }
    frame = submission_rows(
        date(2021, 8, 18), prediction, "H", np.array([100.0, 110.0, 120.0]), np.array([0.7, 0.6, 0.5])
    )
    assert frame.columns == PRED_COLS and frame.height == 96
    assert frame["model"].unique().to_list() == ["B4-H"]
    assert frame["datetime"][0] == "2021.08.18 00:00:00"
    assert frame["datetime"][-1] == "2021.08.18 23:45:00"
    assert frame["M_hat_mean"].unique().to_list() == [46.0]
    assert frame["peak_time_mode"].unique().to_list() == ["11:15"]
    assert "is_missing" not in frame.columns


def test_requirements_are_exported_runtime_pins() -> None:
    lines = (ROOT / "requirements.txt").read_text().splitlines()
    assert lines[0] == "--extra-index-url https://download.pytorch.org/whl/cu126"
    pins = [line.split(";")[0].strip() for line in lines if "==" in line and not line.startswith("#")]
    assert "torch==2.14.0+cu126" in pins
    assert not any(pin.startswith(("pytest==", "requests==")) for pin in pins)


def test_frozen_h_settings_survive_selection_boundary() -> None:
    from gmst.run_outputs import Json, parse_selection

    value: Json = {
        "frozen": True,
        "mode": "development",
        "submitted_model": "B4",
        "submitted_variant": "H",
        "submitted": "B4-H",
        "tau": 10.0,
        "half_life": 60,
        "final_epochs": 7,
        "components": ["H"],
        "thresholds": [100.0, 110.0, 120.0],
        "final_settings": {"lambda_io": 0.1, "M": 8, "lambda_spl": 0.01, "occ_floor": True},
    }
    # Given a fixed H selection, when parsed, then its identity and settings survive.
    selected = parse_selection(value)
    assert selected.variant == "H" and selected.components == ("H",)
    assert selected.epochs == 7 and selected.occ_floor
    assert selected.thresholds == (100.0, 110.0, 120.0)


def test_smoke_selection_cannot_trigger_actual_unseal(tmp_path: Path) -> None:
    import json

    # Given a smoke artifact, when --final is requested, then no final marker is created.
    value = {
        "frozen": True,
        "mode": "smoke",
        "submitted_model": "B4",
        "submitted_variant": "H",
        "submitted": "B4-H",
        "tau": 10.0,
        "half_life": 60,
        "final_epochs": 2,
        "components": ["H"],
        "final_settings": {},
        "thresholds": [100.0, 110.0, 120.0],
    }
    (tmp_path / "selection.json").write_text(json.dumps(value), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "gmst.run_all", "--final", "--out", str(tmp_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "full development" in result.stderr
    assert not (tmp_path / "final_started.json").exists()


def test_smoke_default_output_is_separate() -> None:
    from gmst import RESULTS
    from gmst.run_all import output_directory

    assert output_directory(None, True) != RESULTS
    assert output_directory(None, False) == RESULTS


def test_smoke_refuses_explicit_frozen_development_directory(tmp_path: Path) -> None:
    import json

    (tmp_path / "selection.json").write_text(json.dumps({"mode": "development"}))
    result = subprocess.run(
        [sys.executable, "-m", "gmst.run_all", "--smoke", "--out", str(tmp_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert json.loads((tmp_path / "selection.json").read_text()) == {"mode": "development"}


def test_documented_final_retry_requires_same_selection(tmp_path: Path) -> None:
    import pytest

    from gmst.run_all import record_final, start_final

    # Given a failed attempt, a changed selection and missing reason are rejected.
    record = start_final(tmp_path, "fixed selection", None)
    record.update(status="failed", error="synthetic execution defect")
    record_final(tmp_path, record)
    with pytest.raises(ValueError):
        start_final(tmp_path, "changed selection", "repaired defect")
    with pytest.raises(ValueError):
        start_final(tmp_path, "fixed selection", None)
    retried = start_final(tmp_path, "fixed selection", "repaired execution defect")
    assert retried["attempt"] == 2
    retried["status"] = "complete"
    record_final(tmp_path, retried)
    with pytest.raises(ValueError):
        start_final(tmp_path, "fixed selection", "cannot repeat completed evaluation")


@pytest.mark.parametrize("succeeds", [False, True])
def test_final_attempt_is_audited_without_real_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, succeeds: bool
) -> None:
    import hashlib
    import json

    import polars as pl
    import pytest

    from gmst import features, run_all
    from gmst.contracts import Panel
    from gmst.run_outputs import FrozenRun, Json

    # Given a synthetic final panel, audit each outcome without changing its frozen selection.
    value = {
        "frozen": True,
        "mode": "development",
        "submitted_model": "B4",
        "submitted_variant": "main",
        "submitted": "B4",
        "tau": 10.0,
        "half_life": 60,
        "final_epochs": 2,
        "components": [],
        "final_settings": {},
        "thresholds": [100.0, 100.0, 100.0],
    }
    (tmp_path / "selection.json").write_text(json.dumps(value))
    pl.DataFrame({"model": ["B0", "B0p", "B1", "B4"], "variant": ["main"] * 4, "date": ["2021.08.01"] * 4}).write_csv(
        tmp_path / "oof_days.csv"
    )
    days = [date(2021, 1, 1), date(2021, 1, 2), date(2021, 9, 1)]
    panel: Panel = {
        "dates": days,
        "Y": np.full((3, 96), 100.0),
        "X": {},
        "is_missing": np.zeros((3, 96), dtype=np.bool_),
        "days": pl.DataFrame(
            {
                "date": days,
                "test": ["train", "train", "test"],
                "n_missing": [0, 0, 0],
                "is_copy": [False] * 3,
                "is_suspect": [False] * 3,
            }
        ),
        "op": np.ones(3, dtype=np.int8),
        "hol": np.zeros(3, dtype=np.int8),
        "dtype": np.zeros(3, dtype=np.int8),
        "dow": np.zeros(3, dtype=np.int8),
        "month": np.array([1, 1, 9], dtype=np.int8),
    }
    calls: list[bool] = []

    def synthetic_loader(*, unseal: bool = False) -> Panel:
        calls.append(unseal)
        return panel

    def synthetic_inference(
        panel: Panel, choice: FrozenRun, reference: pl.DataFrame
    ) -> tuple[dict[str, pl.DataFrame], Json]:
        if succeeds:
            return {}, {"B4": {"main": {"test": {"C50": 0.5, "C75": 0.6, "C90": 0.7}}}}
        raise RuntimeError("synthetic execution defect")

    monkeypatch.setattr(features, "load_panel", synthetic_loader)
    monkeypatch.setattr(run_all, "final_tables", synthetic_inference)
    monkeypatch.setattr(run_all, "write_requirements", lambda: True)
    original = (tmp_path / "selection.json").read_bytes()
    if succeeds:
        assert run_all.main(["--final", "--out", str(tmp_path)]) == 0
    else:
        with pytest.raises(RuntimeError, match="synthetic execution defect"):
            run_all.main(["--final", "--out", str(tmp_path)])
    recorded = json.loads((tmp_path / "final_started.json").read_text())
    expected_status = "complete" if succeeds else "failed"
    assert recorded["status"] == expected_status
    assert recorded["attempt"] == 1
    assert calls == [False, True]
    assert json.loads((tmp_path / "final_attempts.jsonl").read_text())["status"] == expected_status
    assert (tmp_path / "selection.json").read_bytes() == original
    assert recorded["selection_sha256"] == hashlib.sha256(original).hexdigest()
    if succeeds:
        stars = json.loads((tmp_path / "final_p_star.json").read_text())
        assert stars["B4"]["main"]["test"] == {"C50": 0.5, "C75": 0.6, "C90": 0.7}
