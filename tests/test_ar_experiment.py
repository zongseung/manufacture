from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import polars as pl
import pytest

from gmst import RESULTS, features


def test_eight_windows_are_prior_only_and_leave_september_sealed() -> None:
    from gmst.ar_experiment import WINDOW_STARTS, experiment_panel

    panel = experiment_panel(features.load_panel())
    assert len(WINDOW_STARTS) == 8
    assert WINDOW_STARTS[0] == date(2021, 7, 1)
    assert WINDOW_STARTS[-1] == date(2021, 8, 18)
    usable = []
    for number, start in enumerate(WINDOW_STARTS, 1):
        fold = f"ar{number}"
        roles = panel["days"][fold].to_list()
        train = [i for i, role in enumerate(roles) if role == "train"]
        validation = [i for i, role in enumerate(roles) if role == "val"]
        assert len(validation) == 14
        assert panel["dates"][validation[0]] == start
        assert panel["dates"][validation[-1]] == start + timedelta(days=13)
        assert all(panel["dates"][i] < start - timedelta(days=1) for i in train)
        assert roles[panel["dates"].index(start - timedelta(days=1))] == "gap"
        train_events = set(panel["days"]["event_id"][train].to_list())
        validation_events = set(panel["days"]["event_id"][validation].to_list())
        assert train_events.isdisjoint(validation_events)
        usable.append(int(np.isfinite(panel["Y"][validation]).any(axis=1).sum()))
    assert usable == [13, 12, 13, 13, 13, 14, 14, 14]
    assert np.isnan(panel["Y"][panel["dates"].index(date(2021, 9, 1)):]).all()


def test_september_window_is_rejected_before_evaluation(monkeypatch: pytest.MonkeyPatch) -> None:
    from gmst import ar_experiment
    from test_hmm import synthetic_panel

    monkeypatch.setattr(ar_experiment, "WINDOW_STARTS", (date(2021, 9, 1),))
    with pytest.raises(ValueError, match="September"):
        ar_experiment.experiment_panel(synthetic_panel())


def test_one_window_evaluates_benchmark_and_four_ar_candidates() -> None:
    from gmst.ar_experiment import ExperimentConfig, evaluate_fold
    from test_hmm import synthetic_panel

    panel = synthetic_panel()
    panel["days"] = panel["days"].with_columns(pl.col("f1").alias("ar8"))
    metrics = evaluate_fold(panel, "ar8", ExperimentConfig(smoke=True, seed=0))
    assert set(metrics["candidate"]) == {"B1", "B3", "B4", "B4-low", "B4-PC"}
    assert metrics["fold"].unique().to_list() == ["ar8"]
    assert metrics["n_slots"].min() > 0
    assert np.isfinite(metrics["mae"].to_numpy()).all()
    assert metrics.filter(pl.col("candidate") == "B3")["phi_mean"][0] == 0
    assert metrics.filter(pl.col("candidate") == "B4-low")["pc_rate"][0] == 0
    assert metrics.filter(pl.col("candidate") == "B4-PC")["pc_rate"][0] > 0
    for threshold in ("C50", "C75", "C90"):
        assert f"brier_{threshold}" in metrics.columns
        assert metrics[f"brier_n_{threshold}"].min() > 0
        assert metrics[f"events_{threshold}"].min() >= 0
        assert metrics[f"calibration_{threshold}"].is_not_null().all()


def test_summary_requires_pc_to_beat_low_start_and_all_seeds() -> None:
    from gmst.ar_experiment import summarize

    rows = [(fold, seed, candidate, mae, mae / 2)
            for fold in ("ar1", "ar2") for seed in (0, 1, 2)
            for candidate, mae in (("B3", 10.0), ("B4", 12.0), ("B4-low", 9.5), ("B4-PC", 9.0))]
    metrics = pl.DataFrame(rows, schema=["fold", "seed", "candidate", "mae", "crps"], orient="row")
    summary, decision = summarize(metrics, required_folds=2)
    assert summary.filter((pl.col("candidate") == "B4-PC") & (pl.col("seed") == 0))["mae_mean"][0] == 9
    assert decision["pc_qualified"]
    assert decision["winners_by_seed"] == {"0": "B4-PC", "1": "B4-PC", "2": "B4-PC"}
    changed = metrics.with_columns(pl.when((pl.col("seed") == 2) & (pl.col("candidate") == "B4-PC"))
                                   .then(10.5).otherwise(pl.col("mae")).alias("mae"))
    _, rejected = summarize(changed, required_folds=2)
    assert not rejected["pc_qualified"]
    assert rejected["winners_by_seed"]["2"] == "B4-low"


def test_b3_wins_an_exact_mae_tie_even_when_pc_has_better_crps() -> None:
    from gmst.ar_experiment import summarize

    metrics = pl.DataFrame([(fold, seed, candidate, mae, crps)
                            for fold in ("ar1", "ar2") for seed in (0, 1, 2)
                            for candidate, mae, crps in (("B3", 9.0, 9.0), ("B4", 12.0, 6.0),
                                                         ("B4-low", 9.5, 5.0), ("B4-PC", 9.0, 1.0))],
                           schema=["fold", "seed", "candidate", "mae", "crps"], orient="row")
    _, decision = summarize(metrics, required_folds=2)
    assert not decision["pc_qualified"]
    assert set(decision["winners_by_seed"].values()) == {"B3"}


def test_smoke_writes_separate_artifacts_without_touching_frozen_final(tmp_path: Path) -> None:
    from gmst.ar_experiment import ExperimentConfig, run

    frozen = RESULTS / "selection.json"
    before = hashlib.sha256(frozen.read_bytes()).hexdigest()
    out = tmp_path / "ar-smoke"
    run(out, ExperimentConfig(smoke=True))
    metrics = pl.read_csv(out / "fold_metrics.csv")
    summary = pl.read_csv(out / "summary.csv")
    decision = json.loads((out / "decision.json").read_text())
    assert metrics.height == summary.height == 5
    assert set(metrics["candidate"]) == {"B1", "B3", "B4", "B4-low", "B4-PC"}
    assert set(metrics["fold"]) == {"ar8"}
    assert decision["status"] == "smoke_only"
    assert not decision["pc_qualified"]
    assert hashlib.sha256(frozen.read_bytes()).hexdigest() == before


def test_experiment_refuses_official_results_and_invalid_tail(tmp_path: Path) -> None:
    from gmst.ar_experiment import ExperimentConfig, run

    with pytest.raises(ValueError, match="official"):
        run(RESULTS, ExperimentConfig(smoke=True))
    with pytest.raises(ValueError, match="PC tail"):
        run(tmp_path / "bad", ExperimentConfig(smoke=True, tail_u=0))
    assert not (tmp_path / "bad").exists()


def test_cli_reports_bad_tail_without_traceback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                capsys: pytest.CaptureFixture[str]) -> None:
    from gmst.ar_experiment import main

    monkeypatch.setattr(sys, "argv", ["ar_experiment", "--smoke", "--u", "0",
                                  "--out", str(tmp_path / "bad")])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert "PC tail" in stderr
    assert "Traceback" not in stderr
