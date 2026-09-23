"""Single entry point; only explicit --final may open the September seal."""
# SIZE_OK: keep the requested CLI/package APIs and sole-unseal execution boundary together.

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

from gmst import DATA, RESULTS, ROOT
from gmst.run_outputs import stage

if TYPE_CHECKING:
    import polars as pl

    from gmst.contracts import Panel
    from gmst.run_outputs import FrozenRun, Json


def write_requirements(path: Path = ROOT / "requirements.txt") -> bool:
    """Export locked runtime dependencies without contacting package indexes."""
    if shutil.which("uv") is None:
        # ponytail: uv가 없으면 커밋된 requirements.txt 유지, 재생성이 필요하면 importlib.metadata 직접 핀 (FR-100)
        return False
    result = subprocess.run(
        [
            "uv",
            "export",
            "--offline",
            "--frozen",
            "--no-hashes",
            "--no-dev",
            "--no-emit-project",
            "--format",
            "requirements.txt",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    path.write_text("--extra-index-url https://download.pytorch.org/whl/cu126\n" + result.stdout, encoding="utf-8")
    return True


def build_package(dest: Path = ROOT / "dist" / "kamp_power_src.zip") -> Path:
    """Package only explicit delivery roots, excluding hidden files and symlinks."""
    files = [ROOT / name for name in ("README.md", "requirements.txt", "pyproject.toml", "uv.lock", "PONYTAIL-DEBT.md")]
    for directory, pattern in (("gmst", "*.py"), ("tests", "*.py"), ("notebooks", "*.ipynb"), ("results", "*")):
        files.extend((ROOT / directory).rglob(pattern))
    files.extend(DATA.glob("*.csv"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as package:
        for path in sorted(set(files)):
            relative = path.relative_to(ROOT)
            if (
                path.is_file()
                and not path.is_symlink()
                and not any(part.startswith(".") or part == "__pycache__" for part in relative.parts)
            ):
                package.write(path, relative.as_posix())
    return dest


def persist(out: Path, tables: dict[str, "pl.DataFrame"], metadata: dict[str, "Json"]) -> None:
    """Only the runner writes result artifacts."""
    from gmst.run_outputs import clean_json

    out.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.write_csv(out / name)
    for name, value in metadata.items():
        (out / name).write_text(
            json.dumps(clean_json(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )


def final_tables(
    panel: "Panel", choice: "FrozenRun", reference: "pl.DataFrame"
) -> tuple[dict[str, "pl.DataFrame"], "Json"]:
    """Produce issued values separately from retrospective observation-mask metrics."""
    import numpy as np
    import polars as pl

    from gmst import baselines, features, hmm
    from gmst import evaluate as ev
    from gmst.contracts import Model
    from gmst.run_outputs import RunConfig, group, pstar_json, submission_rows

    config = RunConfig(choice.smoke, choice.occ_floor)
    fold = "f4" if choice.smoke else "test"
    slots_all, days_all, inner_all = [], [], []

    def evaluate[S](model: Model[S], variant: str = "main") -> dict[str, S]:
        slots, days, states, inner = ev.rolling_origin(model, panel, (fold,), variant)
        slots_all.append(slots)
        days_all.append(days)
        inner_all.append(inner)
        return states

    with stage(13, "final"):
        evaluate(baselines.b0_model())
        evaluate(baselines.b0p_model())
        evaluate(baselines.b1_model(choice.tau, choice.half_life, rounds=config.rounds))
        model = hmm.hmm_model(
            "B4",
            tau=choice.tau,
            half_life=choice.half_life,
            max_epochs=config.epochs,
            final_epochs=choice.epochs,
            N=config.paths,
            rounds=config.rounds,
            occ_floor=choice.occ_floor,
            emission_source="b1" if "H" in choice.components else "backbone",
            decoder="io" if "IO" in choice.components else "constant",
            kan="KAN" in choice.components,
            M=choice.M,
            lambda_io=choice.lambda_io,
            lambda_spl=choice.lambda_spl,
        )
        states = evaluate(model, choice.variant)
        raw_inner = pl.concat(inner_all, how="diagonal_relaxed")
        slots = pl.concat(slots_all, how="diagonal_relaxed")
        days, inner = ev.calibrate_oof(
            pl.concat(days_all, how="diagonal_relaxed"), "platt", raw_inner, ref=None if choice.smoke else reference
        )
        keys = [(str(m), str(v), str(f)) for m, v, f in days.select(ev.KEYS).unique().iter_rows()]
        stars = ev.p_star_table(inner, keys)
    with stage(14, "submission"):
        rows = []
        mask_rows = []
        C = np.asarray(choice.thresholds, dtype=np.float64)
        calibration_rows = group(raw_inner if choice.smoke else reference, "B4", choice.variant)
        calibration = ev.fit_calibrator(calibration_rows, "platt")
        for d in features.role_idx(panel, fold, "val" if choice.smoke else "test"):
            prediction = model["predict"](states[fold], panel, int(d))
            risk = np.array(
                [
                    ev.platt_apply(tuple(calibration[c]["params"]), prediction["risk_raw"][j : j + 1])[0]
                    for j, c in enumerate(ev.CS)
                ]
            )
            frame = submission_rows(panel["dates"][d], prediction, choice.variant, C, np.minimum.accumulate(risk))
            rows.append(frame)
            mask_rows.append(frame.select("datetime").with_columns(pl.Series("is_missing", panel["is_missing"][d])))
        metrics = (
            pl.concat(
                [ev.point_metrics(slots, days, panel), ev.risk_metrics(days, panel, stars)], how="diagonal_relaxed"
            )
            .filter(pl.col("fold") == fold)
            .with_columns(pl.lit("test").alias("fold"))
        )
        return {
            "test_predictions.csv": pl.concat(rows),
            "eval_mask.csv": pl.concat(mask_rows),
            "test_metrics.csv": metrics,
            "final_calibration_status.csv": ev.p_star_status(inner, keys),
        }, pstar_json(stars)


def output_directory(requested: Path | None, smoke: bool) -> Path:
    """Keep pseudo-test artifacts away from a frozen development selection."""
    return requested if requested is not None else ROOT / "results_smoke" if smoke else RESULTS


def start_final(out: Path, selection_text: str, reason: str | None) -> dict[str, "Json"]:
    """Permit documented repairs only after a failed run with identical selection."""
    digest = hashlib.sha256(selection_text.encode()).hexdigest()
    path = out / "final_started.json"
    attempt = 1
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if (
            previous.get("status") != "failed"
            or previous.get("selection_sha256") != digest
            or not reason
            or not reason.strip()
        ):
            raise ValueError("final retry requires failed status, unchanged selection and --retry-reason")
        attempt = int(previous["attempt"]) + 1
    elif reason is not None:
        raise ValueError("--retry-reason requires a previously failed final run")
    record: dict[str, Json] = {
        "status": "started",
        "selection_sha256": digest,
        "attempt": attempt,
        "repair_reason": reason,
    }
    persist(out, {}, {"final_started.json": record})
    return record


def record_final(out: Path, record: dict[str, "Json"]) -> None:
    """Append the final-attempt audit and update its terminal status."""
    persist(out, {}, {"final_started.json": record})
    with (out / "final_attempts.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    """Run development by default; package mode never trains or loads targets."""
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--final", action="store_true")
    modes.add_argument("--no-final", action="store_true")
    modes.add_argument("--package", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--retry-reason")
    parser.add_argument("--occ-floor", action="store_true")
    args = parser.parse_args(argv)
    if args.package:
        print(build_package())
        return 0
    args.out = output_directory(args.out, args.smoke)
    if args.retry_reason is not None and not args.final:
        parser.error("--retry-reason applies only to --final")
    if args.smoke and (args.out / "selection.json").exists():
        previous = json.loads((args.out / "selection.json").read_text(encoding="utf-8"))
        if previous.get("mode") != "smoke":
            parser.error("smoke cannot overwrite a frozen development selection; choose another --out")
    if args.final and not (args.out / "selection.json").is_file():
        parser.error("--final requires a frozen selection.json in --out")
    if not args.final and (args.out / "final_started.json").exists():
        parser.error("a real final run has already started in --out; do not reopen the seal")
    import polars as pl

    from gmst import analysis, features, hmm, preprocess, scenario, splits
    from gmst import evaluate as ev
    from gmst.run_development import development
    from gmst.run_outputs import RunConfig, parse_selection

    out: Path = args.out
    if args.final:
        raw = (out / "selection.json").read_text(encoding="utf-8")
        import numpy as np

        selection_json: dict[str, Json] = json.loads(raw)
        choice = parse_selection(selection_json)
        sealed_panel = features.load_panel()
        C, _ = ev.thresholds(sealed_panel, features.role_idx(sealed_panel, "test", "train"))
        if not np.array_equal(C, np.asarray(choice.thresholds)):
            parser.error("frozen thresholds differ from current pre-September data")
        reference = pl.read_csv(out / "oof_days.csv", infer_schema_length=None)
        if reference.filter(pl.col("date").str.replace_all("-", ".", literal=True) >= "2021.09.01").height:
            parser.error("OOF reference contains sealed-period dates")
        required = {("B0", "main"), ("B0p", "main"), ("B1", "main"), ("B4", choice.variant)}
        if not required <= set(reference.select("model", "variant").iter_rows()):
            parser.error("OOF reference lacks the selected variant or final comparators")
        journal = start_final(out, raw, args.retry_reason)
        try:
            panel = features.load_panel(unseal=True)
            tables, stars = final_tables(panel, choice, reference)
            persist(
                out,
                tables,
                {"final_p_star.json": stars, "final_status.json": {"status": "complete", "mode": "real_final"}},
            )
        except Exception as error:  # CLI audit records unexpected failure before re-raising.
            journal.update(status="failed", error=f"{type(error).__name__}: {error}")
            record_final(out, journal)
            raise
        journal["status"] = "complete"
        record_final(out, journal)
        write_requirements()
        return 0
    for name in (
        "test_predictions.csv",
        "eval_mask.csv",
        "test_metrics.csv",
        "final_p_star.json",
        "final_calibration_status.csv",
        "final_status.json",
    ):
        (out / name).unlink(missing_ok=True)
    with stage(1, "preprocess"):
        preprocess.run(out)
    with stage(2, "splits"):
        splits.run()
    config = RunConfig(args.smoke, args.occ_floor)
    run = development(config)
    tables, panel = run["tables"], run["panel"]
    selection = run["metadata"]["selection.json"]
    choice = parse_selection(selection, allow_smoke=True)
    with stage(10, "state_stability"):
        fold = config.folds[0]
        state = run["states"]["main", fold]
        stability = hmm.state_stability(
            panel,
            fold,
            tau=state["tau"],
            half_life=int(state["half_life"]),
            max_epochs=config.epochs,
            occ_floor=config.occ_floor,
            device=hmm.DEVICE,
        )
        tables["state_stability.csv"] = pl.DataFrame(stability["rows"])
        if stability["collapsed"]:
            print("occupancy collapse (<0.02): rerun development with --occ-floor", flush=True)
    with stage(11, "analysis"):
        pairs = list(dict.fromkeys([("B4", run["variant"]), ("B4", "main"), ("B1", "main")]))
        tables["error_by_condition.csv"] = analysis.error_by_condition(tables["oof_slots.csv"], panel, pairs)
        tables["fn_fp_days.csv"], tables["fn_fp_summary.csv"] = analysis.fn_fp(
            tables["oof_days.csv"], panel, run["stars"], pairs
        )
        tables["leakage_gap.csv"] = analysis.leakage_gap(
            num_boost_round=config.rounds, max_groups=5 if config.smoke else None
        )
        tables["hmm_transitions.csv"] = analysis.transitions(run["states"]["main", config.folds[-1]], panel)
    with stage(12, "scenarios"):
        scenario_states = {fold: state for (variant, fold), state in run["states"].items() if variant == "B"}
        tables["scenarios.csv"], tables["scenarios_month.csv"] = scenario.run_scenarios(
            panel, scenario_states, N=config.paths, max_days=2 if config.smoke else None
        )
    persist(out, tables, run["metadata"])
    if config.smoke:
        final, stars = final_tables(panel, choice, tables["oof_days.csv"])
        persist(
            out,
            final,
            {"final_p_star.json": stars, "final_status.json": {"status": "complete", "mode": "smoke_pseudo_test"}},
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
