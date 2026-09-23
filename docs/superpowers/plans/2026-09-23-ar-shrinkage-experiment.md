# AR Shrinkage Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare B3, original B4, low-initialized B4, and state-wise PC-prior B4 on eight sealed pre-September rolling windows without changing the frozen final run.

**Architecture:** Add two optional AR settings to the existing flat HMM training path. Reuse `rolling_origin`, temporal calibration, and metric functions in a separate CLI that writes experiment artifacts outside official `results/`.

**Tech Stack:** Python 3.13, uv, numpy, torch, polars, pytest; no new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-23-ar-shrinkage-experiment-design.md`

## Global Constraints

- Work in the existing `kamp-core` worktree. Do not commit, stage, push, unseal September, or alter frozen `results/`.
- Preserve every existing B2/B3/B4 default and the official runner behavior.
- Use the original AR(1) PC density for constant positive φ, scaled to the total observed target-slot log likelihood.
- Treat overlapping validation windows as repeated sensitivity checks, not independent evidence.

## Review Focus

- `φ` near zero or one must produce finite prior loss and gradients.
- The same prior is counted once per fit, independently of observed-slot count.
- Validation and calibration labels never enter the fitted state for the same fold.
- The experiment must neither read real September targets nor overwrite official results.
- A one-class risk threshold remains explicit in the output and cannot decide adoption.

---

### Task 1: Optional low φ start and PC prior in the HMM

**Files:** Modify `gmst/hmm_core.py`, `gmst/hmm_training.py`, `gmst/hmm.py`, `gmst/hmm_forecast.py`; test `tests/test_hmm.py`.

**Interfaces:** `CondHMM(..., phi_start=None)`, `pc_rate_for_tail(u, alpha)`, `pc_nlog_prior(phi, rate)`, `hmm_model(..., phi_start=None, pc_rate=0)`; defaults preserve existing models.

- [x] Write tests for default φ, low φ, independent PC density at φ=0.5, tail-rate calibration, finite gradients, and observed-slot normalization of a one-epoch fit. The new tests must fail against current code.
- [x] Run `uv run pytest tests/test_hmm.py -q` and confirm the failures name missing behavior.
- [x] Implement optional initialization and prior through the existing fit path; reject PC with B3 or IO and invalid tail inputs. Keep the prior off by default.
- [x] Run `uv run pytest tests/test_hmm.py -q` and `uv run pytest -q` to green.

### Task 2: Eight strictly prior rolling windows

**Files:** Create `gmst/ar_experiment.py`; test `tests/test_ar_experiment.py`.

**Interfaces:** `experiment_panel(panel)` adds `ar1`…`ar8` role columns by `splits.fold_roles`; `run(out, smoke=False, ...)` evaluates sealed panel only.

- [x] Write a failing test using the real sealed panel: 8 windows, expected usable validation days, prior-only training roles, event purge, no September target exposure.
- [x] Run `uv run pytest tests/test_ar_experiment.py -q` and confirm failure.
- [x] Implement the eight-window panel and model loop with existing `rolling_origin` and `calibrate_oof`; B1 once per fold, B3/B4/B4-low/B4-PC for seeds 0/1/2.
- [x] Run the targeted and full test suites.

### Task 3: Experiment results and CLI

**Files:** Complete `gmst/ar_experiment.py`; extend `tests/test_ar_experiment.py`; update `README.md` with the separate command and interpretation.

**Interfaces:** `fold_metrics.csv`, `summary.csv`, `decision.json`; `--smoke`, `--out`, `--u`, `--alpha`, `--epochs`, `--paths`, `--rounds`.

- [x] Write failing tests for fold aggregation, seed-stability decision, invalid tail/out path, and CSV/JSON creation from a one-window smoke run.
- [x] Run targeted test and confirm its failure.
- [x] Add minimal aggregation and argparse CLI. Keep all writes under the requested non-official output directory.
- [x] Run `uv run pytest -q`, Ruff and basedpyright checks where available. Both static tools were unavailable; Python compilation and the full test suite passed.
- [x] Run `uv run python -m gmst.ar_experiment --help`, a bad-input case, and a real `--smoke` command; inspect its outputs and confirm `results/selection.json` checksum is unchanged.
