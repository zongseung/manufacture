# KAMP Power Forecast (GMST-Power Contest Core v1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Plan override (user instruction, binding):** this plan does **not** contain implementation code. Implementation is written by `fable`-model implementers. Every task gives: exact files, the interface contract (names, signatures, dtypes, column names, file schemas), the algorithm in precise prose/math with spec references, exact values, **complete runnable test code**, exact commands and expected output. Step pattern per task: write the test → run it (fails) → implement to the contract → run it (passes) → run the full suite → commit. The test code in this plan is the acceptance gate: do not weaken, delete or loosen an assertion. If a spec number in a test fails after a faithful implementation, stop and report it to the controller (do not edit the test).

**Goal:** One command (`uv run python -m gmst.run_all`) turns the raw KAMP CSV into repaired 15-min data, a per-date policy table, rolling-origin OOF results for the model ladder (B0, B0′, BB, B1 benchmark, B2, B3, B4 + ablations), calibrated peak-risk checks, a bootstrap gate of the B4 family against the B1 benchmark with a pre-registered improvement loop (B4-H → B4-IO → B4-KAN) and B4-family selection, SCENARIO what-ifs in KRW, error analysis tables, and the sealed 1,344-row test prediction file.

**Architecture:** A flat package `gmst/` of ~10 plain-function modules (no classes except the one `torch.nn.Module` for the HMM). Measurement repair lives in `preprocess.py`, day-level policy in `splits.py`, masking/protocols in `features.py`; every model is a plain dict of functions `{"name", "fit", "predict", ["posthoc"]}` driven by one rolling-origin loop in `evaluate.py`. `run_all.py` is the only writer of `results/` and the only place that unseals the test window. Notebooks only read files.

**Tech Stack:** Python ≥3.13, uv, polars, numpy, torch 2.14.0+cu126 (cuda if available else cpu), lightgbm (native `lgb.train` API), matplotlib (notebooks), pytest (dev).

**Spec:** `tasks/prd-kamp-power-forecast.md` (PRD, primary for tasks: US-001…US-020, FR-1…FR-106, DC1…DC16) + `document/KAMP_GMST_Power_project_proposal_v2.md` (v2: methods, math, A1–A32) + `.sdd/decisions.md` + rulings in `.sdd/progress.md` (later lines win; R1–R11 of the consistency re-review are binding) + `claudedocs/research_copy_vs_outlier_20260922.md` + `.sdd/kepco_tariff_2021.md`. The spec is frozen at commit `a82ea9d` (B1 = benchmark only, pre-registered improvement loop US-021 / FR-107–FR-110). Executors read the PRD and v2 alongside this plan. Where this plan pins a definition the spec leaves open, the pin is marked **[plan pin]** and listed under "Spec issues found".

## Global Constraints

- Work only inside the worktree `WT = /home/user/manufacture_ai/.claude/worktrees/kamp-core`, branch `kamp-core`. Run every command from `WT` with `uv run …`. The first `uv run` creates `WT/.venv` from the uv cache.
- Commits: `git add` **only the files the task lists** (another agent may have uncommitted edits to `document/` and `tasks/` in this worktree — never stage them). Commit messages end **without** `Co-Authored-By` or `Claude-Session` trailers (user rule).
- Dependencies: only `uv add lightgbm` (runtime) and `uv add --dev pytest` (dev). Never add or import `scipy`, `sklearn`/`scikit-learn`, `pandas`, `requests`. Keep the torch `[tool.uv.sources]` / `[[tool.uv.index]]` (pytorch-cu126) blocks unchanged (FR-1, FR-2).
- LightGBM: native API only (`lightgbm.Dataset`, `lightgbm.train`); the sklearn wrapper needs scikit-learn and is forbidden. Params = library defaults + `{"seed": 0, "deterministic": True, "force_col_wise": True, "verbose": -1}`, `num_boost_round=100` unless a smoke/test argument lowers it (FR-5, FR-49).
- ponytail (binding style): the laziest code that works; stdlib → already-installed deps (polars, numpy, torch, matplotlib) first; no class/abstraction without a second use (the model dict protocol has 7+ uses; `CondHMM` is the only class); fewest files; no config files (constants at the top of the module that uses them). Every deliberate shortcut carries `# ponytail: <ceiling>, <upgrade trigger> (<FR/v2 ref>)` on one line; the marker must match the regex `# ?ponytail:\s*[^,]+,\s*\S+` so that `grep -rnE '(#|//) ?ponytail:' gmst tests` finds it (FR-6).
- Seal (FR-28, DC10): `load_panel()` hides the test window (2021-09-01…09-14: all Y and X = NaN) unless `unseal=True`. Inside `gmst/` the literal `unseal=True` appears only in `gmst/run_all.py`, in the final stage, after `selection.json` is written. All design statistics (thresholds, τ, h, K, p*, bin edges, climatology) use only dates ≤ 2021-08-31. Development runs use `run_all --no-final`. Nobody opens or prints `results/test_metrics.csv`/`test_predictions.csv` values before Task 24's single final run; smoke runs write them into a temp dir that tests only schema-check. The sealed window is evaluated for exactly one B4-family candidate (the one selected on f1–f4), plus B0/B0′/B1 for comparison.
- B1 is a benchmark only (user decision): the submitted model is always a Core B4-family candidate. The CI gate against B1 (ΔMAE and Δmean-Brier day-block-bootstrap 95 % upper bounds < 0 on the 54 OOF days) is the success criterion; when it fails, the pre-registered improvement loop ① B4-H → ② B4-IO → ③ B4-KAN runs (Tasks 21–23) until the gate is met or 2026-10-03, and every candidate (run or not) is reported with its count.
- Masking is applied in three places at once: targets, lag/rolling inputs, HMM warm-up (DC1, DC2, DC14, [수정-5]). The loader does it once by writing NaN into `Y` (and `생산량` for suspect days); downstream code must treat NaN as "unknown" and never fill it.
- Inner validation (ruling R10, FR-33/55/61/68): every fold's tuning (backbone (τ,h) pooled over f1–f4 inner windows, HMM early stopping, p*) uses the **inner window** = the last 7 calendar days of that fold's `train` role; after choosing, the fold model is **refit on the full train window** and only that refit predicts the fold's validation days. The fold's own validation days are never a selection criterion. Calibration stays leave-one-fold-out (FR-54).
- Device (FR-4): `gmst/hmm.py` defines `DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")`; one GPU at most. `uv run pytest -q` must pass on the GPU machine and with `CUDA_VISIBLE_DEVICES=""`. Tests that compare bitwise pass `device="cpu"`.
- Determinism (FR-5): numpy `np.random.default_rng(seed)`, `torch.manual_seed(seed)`, seed 0 by default; MC draws use a per-date `torch.Generator` seeded with `int(date.strftime("%Y%m%d"))`.
- Paths: `pathlib` only; the data folder name contains a space and Hangul. Raw CSV has a BOM and **must** be read with `pl.read_csv(path, infer_schema_length=None)` (default inference fails on `강수량` = `0.2`, verified).
- Formats: dates in CSV outputs `%Y.%m.%d`; datetimes `%Y.%m.%d %H:%M:%S`; booleans `true`/`false` (polars default); JSON dates ISO `YYYY-MM-DD`; NaN written to JSON as `null`.
- Model IDs: `B0`, `B0p`, `BB`, `B1`, `B2`, `B3`, `B4`. Variants: `main`, `copies`, `suspect`, `protoA`, `K2`, `K4`, `AWstar`, `B` (FR-73, FR-95), `re` (diagnostic, FR-75), plus the B4 improvement-loop variants `H`, `IO`, `KAN` (model `B4`; submission model IDs `B4`, `B4-H`, `B4-IO`, `B4-KAN`; Tasks 21–23, US-021). Folds: `f1`…`f4`, `pooled`, `test`. Strata: `all`, `op`, `nonop`.
- No network, no credentials, no `.env` reading, no affiliation/logo text anywhere (FR-3, FR-7).
- Every task ends with `uv run pytest -q` passing (all earlier tests included).

## File Structure

| Path | Responsibility | Task |
|---|---|---|
| `pyproject.toml` (modify) | add lightgbm, dev pytest, `[tool.pytest.ini_options]` | 1 |
| `gmst/__init__.py` | `ROOT`, `DATA`, `RESULTS` only | 1 |
| `gmst/preprocess.py` | raw hourly CSV → repaired 15-min CSV + `data_audit.json` | 2 |
| `gmst/splits.py` | per-date policy table (copies, suspect, operating, holiday, fold roles) | 3 |
| `gmst/features.py` | `load_panel` (masking, seal), protocols, calendar, B1 row builders | 4, 8 |
| `gmst/evaluate.py` | metrics, thresholds, rolling-origin loop, calibration, p*, climatology, risk check, bootstrap, DM, selection | 5, 6, 9, 10 |
| `gmst/baselines.py` | B0, B0′, lag-7 table, B1 LightGBM | 6, 8 |
| `gmst/backbone.py` | cyclic-RW2 backbone, BB model, (τ,h) selection, as-of backbone | 7 |
| `gmst/hmm.py` | conditional HMM, forward/backward, training, MC forecast, analytic risk, B2/B3/B4 factory, transition table, state stability | 11, 12, 14 |
| `gmst/scenario.py` | KEPCO 2021 tariff, ratchet, schedule transforms, SCENARIO what-ifs | 15, 16 |
| `gmst/analysis.py` | error by condition, FN/FP tables, augmentation-leakage gap | 17 |
| `gmst/run_all.py` | one-command runner, smoke, final unseal, submission files, requirements, package | 18, 19 |
| `tests/test_style.py` | ponytail marker format, banned imports, unseal location, deps, ledger/README checks | 1, 22 |
| `tests/test_preprocess.py`, `test_splits.py`, `test_features.py`, `test_evaluate.py`, `test_baselines.py`, `test_backbone.py`, `test_hmm.py`, `test_leakage.py`, `test_state_stability.py`, `test_scenario.py`, `test_analysis.py`, `test_run_all.py`, `test_results.py`, `test_notebooks.py` | tests per module / cross-cutting | 2–25 |
| `notebooks/01_preprocess.ipynb`, `02_eda.ipynb` (modify), `03_results.ipynb` (create) | display-only viewers | 21 |
| `requirements.txt`, `.gitignore` (modify), `README.md` (fill), `PONYTAIL-DEBT.md` | submission artifacts | 19, 26 |
| `results/` | written only by `run_all` (dev runs Tasks 20–23, committed after the final run in Task 24) | 20–24 |

`main.py` stays untouched (PRD non-goal).

## Shared contracts (read before any task)

**Panel dict** (returned by `features.load_panel`, consumed everywhere). Model code may read only these keys: `dates` (list of 257 `datetime.date`, 2021-01-01…2021-09-14), `Y` (257×96 float64, NaN = masked/missing/sealed), `X` (dict with keys `생산량`, `기온`, `풍속`, `습도`, `강수량_증분`, each 257×96 float64, hourly value repeated over its 4 slots, NaN when sealed; `생산량` NaN on suspect days), `is_missing` (257×96 bool, the outage flag), `days` (polars DataFrame = the 17-column day table, `date` as `pl.Date`), `op` (257 int8, true `is_operating`), `hol` (257 int8), `dtype` (257 int8: 0 `wk`, 1 `sat`, 2 `sun`), `dow` (257 int8, Monday = 0), `month` (257 int8).

**Model dict protocol** (rolling-origin input): `{"name": str, "fit": fit(panel, fold, C) -> state, "predict": predict(state, panel, d) -> pred, "posthoc": optional posthoc(state, panel, d) -> (filt, smooth)}`. `fold` ∈ {`f1`,`f2`,`f3`,`f4`,`test`}; `C` = np.ndarray (3,) thresholds (C50, C75, C90) of that fold; `d` = int day index. A state may carry `"inner_state"` (a state accepted by the same `predict`) holding the model refit-free inner-window fit used for inner-day predictions (p*). `predict` must use only `Y[:d]`, `X[:d]`, calendar of `d`, and the protocol-allowed exogenous values of day `d` (FR-33).

**Prediction dict** (FR-34): `y_mean` (96,), `y_median` (96,), `q` (19×96 or `None`), `paths` (N×96 or `None`), `M_hat_median` (float), `M_hat_mean` (float), `peak_time_mode` (int 0–95), `risk_raw` (np.ndarray (3,) for C50/C75/C90). Extra keys are allowed (B1 adds `Mq`).

**Quantile grid** (FR-35): `TAUS = 0.05, 0.10, …, 0.95` (19); column names `q05, q10, …, q95`.

**Schedule (ruling R11 + user decision on the loop):** D = 2026-09-23. Target: Tasks 1–4 09-23, 5–7 09-24, 8 09-25, 9–10 09-26, 11 09-27, 12 09-28, 13–14 09-29, 15–16 09-30, 17–19 10-01, 20 (dev run + gate) 10-01/10-02, improvement loop Tasks 21–23 only while the gate fails and never after **2026-10-03**, 24 (single final run) 10-04, 25–26 and the report 10-04…10-06; buffer 10-07…10-08 (deadline 2026-10-08 23:59).

**Cut order if late (ruling R3, PRD §5):** ① K=4 ablation (variant `K4`) ② isotonic calibration columns/rows. Cut items stay in the tables as "미실행". Loop variants not reached by 10-03 stay in `selection.json["candidates"]` as `not_implemented`. Never cut: the B1-benchmark gate and B4-family selection, the sealed test, the leakage test (Task 13), Ch.3/Ch.4 analysis, the oracle-weather ablation. B5 (encoder) is a non-goal — no task.

---

### Task 1: Dependencies, package skeleton, style gate

**Files:**
- Modify: `pyproject.toml`, `uv.lock` (via `uv add`)
- Create: `gmst/__init__.py`, `tests/test_style.py`

**Interfaces:**
- Produces: `gmst.ROOT: Path` (= `WT`), `gmst.DATA: Path` (= `ROOT / "5. 자원 최적화 AI 데이터셋"`), `gmst.RESULTS: Path` (= `ROOT / "results"`). Nothing else public in `gmst/__init__.py`.
- `pyproject.toml` gains `lightgbm` in `[project].dependencies`, `pytest` in `[dependency-groups].dev`, and
  ```toml
  [tool.pytest.ini_options]
  pythonpath = ["."]
  testpaths = ["tests"]
  ```

- [ ] **Step 1: Write the failing test** — create `tests/test_style.py`:

```python
import re
import tomllib
from pathlib import Path

import gmst

ROOT = Path(__file__).resolve().parents[1]
MARK = re.compile(r"#\s*ponytail:")
FORMAT = re.compile(r"# ?ponytail:\s*[^,]+,\s*\S+")
BANNED = re.compile(r"^\s*(import|from)\s+(scipy|sklearn|pandas|requests)\b")
ENV = re.compile(r"\.env\b")


def _py(*dirs):
    return sorted(p for d in dirs if (ROOT / d).exists() for p in (ROOT / d).rglob("*.py"))


def _names(deps):
    return {re.split(r"[<>=!~\[ ;]", d, maxsplit=1)[0].lower() for d in deps}


def test_paths():
    assert gmst.ROOT == ROOT
    assert gmst.DATA.name == "5. 자원 최적화 AI 데이터셋"
    assert gmst.RESULTS == gmst.ROOT / "results"
    paths = {k for k, v in vars(gmst).items() if not k.startswith("_") and isinstance(v, Path)}
    assert paths == {"ROOT", "DATA", "RESULTS"}


def test_ponytail_markers_have_ceiling_and_trigger():
    bad = [f"{p.relative_to(ROOT)}:{i}: {line.strip()}"
           for p in _py("gmst", "tests")
           for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
           if MARK.search(line) and not FORMAT.search(line)]
    assert not bad, "\n".join(bad)


def test_no_banned_imports_or_env():
    for p in _py("gmst"):
        text = p.read_text(encoding="utf-8")
        assert not any(BANNED.match(line) for line in text.splitlines()), p
        assert not ENV.search(text), p


def test_unseal_only_in_run_all():
    hits = {p.name for p in _py("gmst") if "unseal=True" in p.read_text(encoding="utf-8")}
    assert hits <= {"run_all.py"}, hits


def test_dependencies():
    pp = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    names = _names(pp["project"]["dependencies"])
    assert "lightgbm" in names
    assert not names & {"requests", "scipy", "scikit-learn", "sklearn", "pandas", "pytest"}
    assert "pytest" in _names(pp["dependency-groups"]["dev"])
    assert pp["tool"]["pytest"]["ini_options"] == {"pythonpath": ["."], "testpaths": ["tests"]}
    assert pp["tool"]["uv"]["sources"]["torch"] == {"index": "pytorch-cu126"}
    assert pp["tool"]["uv"]["index"][0]["url"] == "https://download.pytorch.org/whl/cu126"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run --with pytest pytest tests/test_style.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'gmst'` (pytest is not a dependency yet, hence `--with`).

- [ ] **Step 3: Implement**
  1. `uv add lightgbm` then `uv add --dev pytest` (exactly these two commands; do not touch the torch source/index blocks).
  2. Append the `[tool.pytest.ini_options]` block above to `pyproject.toml`.
  3. Create `gmst/__init__.py` defining exactly `ROOT = Path(__file__).resolve().parents[1]`, `DATA`, `RESULTS` as in Interfaces (import `Path` under a private alias or `from pathlib import Path` — `Path` is a class, not a `Path` instance, so the test ignores it).

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_style.py -q` → `5 passed`.
Run: `uv run python -c "import lightgbm, torch, polars, gmst; print(torch.__version__, lightgbm.__version__)"` → prints `2.14.0+cu126 <lightgbm version>`.

- [ ] **Step 5: Full suite and commit**

Run: `uv run pytest -q` → all passed.
```bash
git add pyproject.toml uv.lock gmst/__init__.py tests/test_style.py
git commit -m "feat: gmst package skeleton, lightgbm + dev pytest, style gate"
```

---

### Task 2: Preprocessing script (measurement repair)

**Files:**
- Create: `gmst/preprocess.py`, `tests/test_preprocess.py`
- Writes (by `run()`): `DATA/okm_15min_2021.csv` (replaces the notebook output, tracked in git), `<out>/data_audit.json`

**Interfaces:**
- Consumes: `gmst.DATA`, `gmst.RESULTS`.
- Produces (module `gmst.preprocess`):
  - `RAW = DATA / "okm_augumented_2021.csv"`, `OUT = DATA / "okm_15min_2021.csv"`
  - `COLS = ["datetime", "전력", "생산량", "기온", "풍속", "습도", "강수량_증분", "전기요금(계절)", "인건비", "is_missing"]`
  - `read_raw() -> pl.DataFrame` — raw file in file order plus an `hour` column (Int64, row position within its `날짜`, 0–23).
  - `precip_increment(c: np.ndarray) -> np.ndarray` — hourly cumulative series (NaN = missing) → hourly increment.
  - `build_15min() -> pl.DataFrame` — no file writes; columns `COLS` in order; `datetime` dtype `pl.Datetime`; `전력` Float64 with nulls at missing points; other numeric columns Float64; `is_missing` Boolean; sorted by datetime; 24,672 rows.
  - `audit() -> dict` — JSON-serializable audit (keys below).
  - `run(out: Path = RESULTS) -> None` — writes `OUT` (datetime formatted `%Y.%m.%d %H:%M:%S`) and `out / "data_audit.json"` (UTF-8, `ensure_ascii=False`); creates `out` if missing.
  - `if __name__ == "__main__": run()` (so `uv run python -m gmst.preprocess` works, FR-3).

**Algorithm (FR-8…FR-15, DC3–DC6, DC15):**
1. `read_raw()`: `pl.read_csv(RAW, infer_schema_length=None)` (BOM is stripped by polars; first column is `날짜`, Int64 like `20210101`). `hour` = `pl.int_range(pl.len()).over("날짜")` in file order (FR-9: the `시간` column is overwritten on 07-13/07-15; position is authoritative).
2. Hourly series in time order (6,168 rows). `강수량` → `precip_increment` (FR-13): walk forward keeping `prev` = last non-NaN value seen; for each hour `t`: if `c_t` is NaN → `inc_t = 0` (prev unchanged); elif no `prev` yet → `inc_t = c_t`; else `inc_t = c_t − prev`, and if `inc_t < 0` → `inc_t = c_t` (reset); then `prev = c_t`. The series is continuous across midnight (never reset at day boundaries by rule).
3. `풍속`: linear interpolation over the hourly index (`np.interp` of NaN positions between neighbouring non-NaN hours) (FR-12, DC4).
4. Wide → long (FR-10): value columns `15분, 30분, 45분, 60분` → k = 0, 1, 2, 3; `datetime = 날짜 date + hour h + 15·k min` (interval-start labels). Hourly covariates (`생산량, 기온, 풍속, 습도, 강수량_증분, 전기요금(계절), 인건비`) are repeated to all 4 slots (not divided).
5. `전력 == 0` → null and `is_missing = True`; else `is_missing = False` (FR-11, DC3).
6. Drop `공장인원, 평균, day, d, m, 날짜, 시간` (FR-14); cast numerics to Float64; select `COLS`.
7. `audit()` keys and meanings (values checked by the test): `n_rows_raw` (6168), `n_rows_15min` (24672), `n_days` (257), `zero_points` (count of raw power values == 0, 74), `zero_by_day` ({ISO date: count}), `wind_null_hours` (raw hourly rows with null `풍속`, 3), `precip_null_hours` (1), `time_col_mismatch_rows` (rows where `시간 != hour`, 48), `time_col_mismatch_dates` (sorted ISO dates), `공장인원_max_abs_err` (max |공장인원 − 생산량/S| over rows with S > 0 and non-null 공장인원, S = sum of the four power values), `공장인원_n` (that row count, 6151), `평균_match` (rows where `평균 == floor(mean4 + 0.5)`, 6168).

- [ ] **Step 1: Write the failing test** — create `tests/test_preprocess.py`:

```python
import hashlib
import json
from datetime import date, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from gmst import preprocess as pp

COLS = ["datetime", "전력", "생산량", "기온", "풍속", "습도", "강수량_증분", "전기요금(계절)", "인건비", "is_missing"]


@pytest.fixture(scope="module")
def df():
    return pp.build_15min()


def test_schema(df):
    assert df.columns == COLS
    assert df.height == 24672
    assert df["datetime"].dtype == pl.Datetime
    assert df["datetime"][0] == datetime(2021, 1, 1, 0, 0)
    assert df["datetime"][-1] == datetime(2021, 9, 14, 23, 45)
    assert df["is_missing"].dtype == pl.Boolean


def test_grid(df):
    dts = df["datetime"]
    assert dts.n_unique() == df.height
    assert dts.diff().drop_nulls().unique().to_list() == [timedelta(minutes=15)]
    per_day = df.group_by(pl.col("datetime").dt.date().alias("d")).len()
    assert per_day.height == 257 and per_day["len"].unique().to_list() == [96]


def test_zero_power(df):
    assert df["전력"].null_count() == 74
    assert df["is_missing"].sum() == 74
    assert (df["전력"].is_null() == df["is_missing"]).all()
    by_day = df.filter(pl.col("is_missing")).group_by(pl.col("datetime").dt.date().alias("d")).len().sort("d")
    assert dict(zip(by_day["d"].to_list(), by_day["len"].to_list())) == {
        date(2021, 8, 28): 26, date(2021, 8, 29): 46, date(2021, 9, 8): 2}
    assert df["전력"].drop_nulls().min() > 0
    assert df.filter(pl.col("is_missing"))["datetime"][0] == datetime(2021, 8, 28, 17, 30)


def test_wind_interp(df):
    raw = pp.read_raw()
    assert df["풍속"].null_count() == 0
    hrs = raw.filter(pl.col("날짜") == 20210601).sort("hour")["풍속"].to_list()
    v0, v3 = hrs[0], hrs[3]
    got = df.filter(pl.col("datetime").dt.date() == date(2021, 6, 1))["풍속"].to_list()
    assert got[4:8] == pytest.approx([v0 + (v3 - v0) / 3] * 4)
    assert got[8:12] == pytest.approx([v0 + 2 * (v3 - v0) / 3] * 4)


def test_precip_increment_synthetic():
    assert pp.precip_increment(np.array([0.0, 0.5, 1.0, 1.0, 0.2])) == pytest.approx([0.0, 0.5, 0.5, 0.0, 0.2])
    assert pp.precip_increment(np.array([0.3, np.nan, 0.5, 0.1])) == pytest.approx([0.3, 0.0, 0.2, 0.1])


def test_precip_increment_real(df):
    s = df["강수량_증분"]
    assert s.null_count() == 0 and s.min() >= 0
    jan24 = df.filter(pl.col("datetime").dt.date() == date(2021, 1, 24))["강수량_증분"].to_list()
    assert jan24[:4] == [0.0, 0.0, 0.0, 0.0]


def test_leakage_audit(df):
    a = pp.audit()
    assert "공장인원" not in df.columns and "평균" not in df.columns
    assert (a["n_rows_raw"], a["n_rows_15min"], a["n_days"]) == (6168, 24672, 257)
    assert a["zero_points"] == 74
    assert a["zero_by_day"] == {"2021-08-28": 26, "2021-08-29": 46, "2021-09-08": 2}
    assert (a["wind_null_hours"], a["precip_null_hours"]) == (3, 1)
    assert a["time_col_mismatch_rows"] == 48
    assert a["time_col_mismatch_dates"] == ["2021-07-13", "2021-07-15"]
    assert a["공장인원_max_abs_err"] <= 4.9e-9
    assert a["공장인원_n"] == 6151
    assert a["평균_match"] == 6168
    json.dumps(a, ensure_ascii=False)


def test_run_writes_and_keeps_raw(tmp_path):
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    before = sha(pp.RAW)
    pp.run(out=tmp_path)
    assert sha(pp.RAW) == before
    lines = pp.OUT.read_text(encoding="utf-8").splitlines()
    assert lines[0] == ",".join(COLS)
    assert lines[1].startswith("2021.01.01 00:00:00,")
    assert lines[-1].startswith("2021.09.14 23:45:00,")
    assert len(lines) == 24673
    a = json.loads((tmp_path / "data_audit.json").read_text(encoding="utf-8"))
    assert a["zero_points"] == 74
```

- [ ] **Step 2: Run to verify it fails** — `uv run pytest tests/test_preprocess.py -q` → `ModuleNotFoundError: No module named 'gmst.preprocess'`.
- [ ] **Step 3: Implement** `gmst/preprocess.py` to the contract and algorithm above.
- [ ] **Step 4: Run to verify it passes** — `uv run pytest tests/test_preprocess.py -q` → `8 passed`. Then `uv run python -m gmst.preprocess` (exit 0; rewrites `DATA/okm_15min_2021.csv` and `results/data_audit.json`).
- [ ] **Step 5: Full suite and commit** — `uv run pytest -q`, then
```bash
git add gmst/preprocess.py tests/test_preprocess.py "5. 자원 최적화 AI 데이터셋/okm_15min_2021.csv"
git commit -m "feat: preprocessing script replaces notebook 15-min output (zeros masked, wind interp, precip increment, leakage audit)"
```
(`results/` is not committed until Task 24.)

---

### Task 3: Day table and fold roles (splits)

**Files:**
- Create: `gmst/splits.py`, `tests/test_splits.py`
- Writes (by `run()`): `DATA/okm_cv_splits_2021.csv` (replaces the notebook output, tracked)

**Interfaces:**
- Consumes: `preprocess.build_15min()`.
- Produces (module `gmst.splits`):
  - `HOLIDAYS_2021: tuple[date, ...]` = (2021-01-01, 02-11, 02-12, 02-13, 03-01, 05-05, 05-19, 08-16) with a source comment "한국천문연구원 특일정보" and the marker `# ponytail: 2021 공휴일 하드코딩, 다른 연도 데이터가 오면 특일정보 목록 교체 (FR-21)`.
  - `SUSPECT = (date(2021, 7, 13), date(2021, 7, 15))`
  - `FOLDS: dict[str, tuple[date, date]]` = `f1` (07-07, 07-20), `f2` (07-21, 08-03), `f3` (08-04, 08-17), `f4` (08-18, 08-31), `test` (09-01, 09-14).
  - `EDITED = (date(2021,1,1), date(2021,1,9), date(2021,1,10), date(2021,1,16), date(2021,3,7), date(2021,3,21), date(2021,3,28))`
  - `COLUMNS = ["date", "event_id", "dup_size", "is_copy", "copy_kind", "copy_of", "is_suspect", "is_operating", "is_holiday", "daytype", "n_missing", "anomaly", "f1", "f2", "f3", "f4", "test"]`
  - `OUT = DATA / "okm_cv_splits_2021.csv"`
  - `active_same(a: np.ndarray, b: np.ndarray) -> int` — number of slots with `a == b` and `a > 40` (NaN never equal).
  - `fold_roles(dates: list[date], events: list[str], start: date, end: date, val_label: str) -> list[str]`
  - `build_days(df15: pl.DataFrame) -> pl.DataFrame` — 257 rows, `COLUMNS` order; dtypes: `date` pl.Date, `event_id` Utf8, `dup_size` Int64, booleans pl.Boolean, `copy_kind` Utf8 ∈ {"", "exact", "edited"}, `copy_of` Utf8 (`"%Y.%m.%d"` or ""), `daytype` Utf8 ∈ {wk, sat, sun}, `n_missing` Int64, roles Utf8 ∈ {train, gap, val, test, purged, ""}.
  - `run() -> pl.DataFrame` — `build_days(preprocess.build_15min())`, writes `OUT` with `date_format="%Y.%m.%d"`; `__main__` calls `run()`.

**Algorithm (FR-16…FR-26, DC1, DC2, DC9, DC12–DC14):**
1. Power matrix `P` (257×96) from `df15` (null → NaN). `event_id`: group days whose 96-vector is identical treating NaN as a value (e.g. key = `tuple(np.nan_to_num(row, nan=-1))`); order groups by their earliest date → `E000`, `E001`, … (142 groups); `dup_size` = group size.
2. Exact-group original (FR-17): original = earliest member, **except** when that member has ≥ 48 slots with power > 40 **and** its daily production sum (sum of the 96 repeated hourly values / 4) is 0 **and** another member has production > 0 → original = the earliest member with production > 0. Non-original members: `is_copy=True`, `copy_kind="exact"`, `copy_of` = original date. (Only the 01-24 group swaps: 01-24 → copy of 02-24.)
3. Edited copies (FR-19): candidates = days with `dup_size == 1`. For each candidate `i`, `n_j = active_same(P[i], P[j])` for all `j ≠ i`; if `max_j n_j ≥ 20` → `is_copy=True`, `copy_kind="edited"`, `copy_of` = the argmax partner, ties broken by (a) partner not flagged `is_copy` after step 2 first, then (b) earliest date (ruling R2). No transitive union; `event_id` unchanged. Expected result (verified on ≤ 08-31): 01-01→01-02 (27), 01-09→02-09 (50), 01-10→02-10 (48), 01-16→02-16 (33), 03-07→01-07 (67), 03-21→01-21 (64), 03-28→01-28 (64).
4. `build_days` asserts (v2 §15.3): the set of `edited` dates == `EDITED`; 01-24 is `exact` with `copy_of == "2021.02.24"` and 02-24 is not a copy; no `edited` date ≥ 2021-07-01.
5. `is_suspect` = date ∈ `SUSPECT`; `is_operating` = daily production sum > 0 or `is_suspect` (FR-20); `is_holiday` = date ∈ `HOLIDAYS_2021`; `daytype` = `wk` (Mon–Fri) / `sat` / `sun`; `n_missing` = count of `is_missing` in the day; `anomaly` = `is_suspect or n_missing > 0` (FR-24).
6. `fold_roles(dates, events, start, end, val_label)`: day in [start, end] → `val_label`; day == start − 1 → `gap`; day < start − 1 → `train`, unless its event is in the event set of the window days → `purged`; day > end → `""`. For fold `test`, `val_label = "test"`; others `"val"`. Comment in code: the gap day is excluded from targets but used as an input for the first validation day (FR-25).

- [ ] **Step 1: Write the failing test** — create `tests/test_splits.py`:

```python
import itertools
from datetime import date

import numpy as np
import polars as pl
import pytest

from gmst import preprocess as pp
from gmst import splits as sp

COLS = ["date", "event_id", "dup_size", "is_copy", "copy_kind", "copy_of", "is_suspect", "is_operating",
        "is_holiday", "daytype", "n_missing", "anomaly", "f1", "f2", "f3", "f4", "test"]
EDITED = [date(2021, 1, 1), date(2021, 1, 9), date(2021, 1, 10), date(2021, 1, 16),
          date(2021, 3, 7), date(2021, 3, 21), date(2021, 3, 28)]


@pytest.fixture(scope="module")
def df15():
    return pp.build_15min()


@pytest.fixture(scope="module")
def days(df15):
    return sp.build_days(df15)


def row(days, d):
    return days.filter(pl.col("date") == d).row(0, named=True)


def power(df15):
    return df15.sort("datetime")["전력"].to_numpy().astype(float).reshape(257, 96)


def test_schema(days):
    assert days.columns == COLS
    assert sp.COLUMNS == COLS
    assert days.height == 257
    assert days["date"].to_list() == sorted(days["date"].to_list())
    assert set(days["copy_kind"].unique().to_list()) <= {"", "exact", "edited"}


def test_copy_days(days):
    assert (days["copy_kind"] == "exact").sum() == 115
    assert days["is_copy"].sum() == 122
    assert days["event_id"].n_unique() == 142
    multi = days.filter(pl.col("dup_size") > 1)
    assert multi["event_id"].n_unique() == 45 and multi.height == 160
    a, b, c = (row(days, date(2021, m, 1)) for m in (2, 3, 6))
    assert a["event_id"] == b["event_id"] == c["event_id"]
    assert (a["is_copy"], b["is_copy"], c["is_copy"]) == (False, True, True)
    assert b["copy_of"] == "2021.02.01"
    r24, r224 = row(days, date(2021, 1, 24)), row(days, date(2021, 2, 24))
    assert (r24["is_copy"], r24["copy_kind"], r24["copy_of"]) == (True, "exact", "2021.02.24")
    assert r224["is_copy"] is False
    assert row(days, date(2021, 7, 28))["is_copy"] is False
    assert row(days, date(2021, 7, 30))["copy_kind"] == "exact"
    moved = [g.sort("date")["date"][0] for _, g in multi.group_by("event_id") if g.sort("date")["is_copy"][0]]
    assert moved == [date(2021, 1, 24)]


def test_edited_copies(days, df15):
    P = power(df15)
    dates = days["date"].to_list()
    idx = {d: i for i, d in enumerate(dates)}
    ed = days.filter(pl.col("copy_kind") == "edited")
    assert ed["date"].to_list() == EDITED
    for r in ed.iter_rows(named=True):
        i = idx[r["date"]]
        y, m, d = map(int, r["copy_of"].split("."))
        best = sp.active_same(P[i], P[idx[date(y, m, d)]])
        assert best >= 20 and r["is_copy"] is True
        assert all(best >= sp.active_same(P[i], P[k]) for k in range(len(dates)) if k != i)
    late = days.filter(pl.col("date") >= date(2021, 7, 1))
    assert late.filter(pl.col("is_copy"))["date"].to_list() == [date(2021, 7, 30)]
    jul_aug = [idx[d] for d in dates if date(2021, 7, 1) <= d <= date(2021, 8, 31) and not row(days, d)["is_copy"]]
    assert len(jul_aug) == 61
    assert max(sp.active_same(P[i], P[j]) for i, j in itertools.combinations(jul_aug, 2)) == 9


def test_suspect(days):
    s = days.filter(pl.col("is_suspect"))
    assert s["date"].to_list() == [date(2021, 7, 13), date(2021, 7, 15)]
    assert s["is_operating"].to_list() == [True, True]


def test_operating(days):
    assert (~days["is_operating"]).sum() == 62
    assert days["is_operating"].sum() == 195


def test_holiday_daytype(days):
    assert len(sp.HOLIDAYS_2021) == 8
    assert days.filter(pl.col("is_holiday"))["date"].to_list() == list(sp.HOLIDAYS_2021)
    assert dict(days.group_by("daytype").len().iter_rows()) == {"wk": 183, "sat": 37, "sun": 37}


@pytest.mark.parametrize("fold,train,gap_day,blank", [
    ("f1", 186, date(2021, 7, 6), 56), ("f2", 200, date(2021, 7, 20), 42),
    ("f3", 214, date(2021, 8, 3), 28), ("f4", 228, date(2021, 8, 17), 14),
    ("test", 242, date(2021, 8, 31), 0)])
def test_roles(days, fold, train, gap_day, blank):
    col = days[fold]
    val_label = "test" if fold == "test" else "val"
    assert (col == "train").sum() == train
    assert days.filter(pl.col(fold) == "gap")["date"].to_list() == [gap_day]
    assert (col == val_label).sum() == 14
    assert (col == "").sum() == blank
    assert (col == "purged").sum() == 0
    assert days.filter(pl.col(fold) == val_label)["date"][0] == sp.FOLDS[fold][0]


def test_purge_synthetic():
    ds = [date(2021, 7, d) for d in range(1, 11)]
    ev = ["E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E1", "E9"]
    assert sp.fold_roles(ds, ev, date(2021, 7, 8), date(2021, 7, 9), "val") == [
        "train", "purged", "train", "train", "train", "train", "gap", "val", "val", ""]


def test_anomaly_nmissing(days):
    assert days.filter(pl.col("anomaly"))["date"].to_list() == [
        date(2021, 7, 13), date(2021, 7, 15), date(2021, 8, 28), date(2021, 8, 29), date(2021, 9, 8)]
    nm = {d: n for d, n in days.select("date", "n_missing").iter_rows() if n}
    assert nm == {date(2021, 8, 28): 26, date(2021, 8, 29): 46, date(2021, 9, 8): 2}


def test_run_writes_csv():
    sp.run()
    lines = sp.OUT.read_text(encoding="utf-8").splitlines()
    assert lines[0] == ",".join(COLS)
    assert len(lines) == 258
    assert lines[1].split(",")[:6] == ["2021.01.01", "E000", "1", "true", "edited", "2021.01.02"]
```

- [ ] **Step 2: Run to verify it fails** — `uv run pytest tests/test_splits.py -q` → `ModuleNotFoundError: No module named 'gmst.splits'`.
- [ ] **Step 3: Implement** `gmst/splits.py` to the contract and algorithm above.
- [ ] **Step 4: Run to verify it passes** — `uv run pytest tests/test_splits.py -q` → `14 passed`; `uv run python -m gmst.splits` exits 0.
- [ ] **Step 5: Full suite and commit**
```bash
git add gmst/splits.py tests/test_splits.py "5. 자원 최적화 AI 데이터셋/okm_cv_splits_2021.csv"
git commit -m "feat: per-date policy table with copy/edited-copy rule, suspect, operating, holidays, fold roles"
```

---

### Task 4: Loader — masking, protocols, seal

**Files:**
- Create: `gmst/features.py`, `tests/test_features.py`

**Interfaces:**
- Consumes: `preprocess.OUT`, `splits.OUT` (both CSVs must exist: run `uv run python -m gmst.preprocess && uv run python -m gmst.splits` once if missing).
- Produces (module `gmst.features`):
  - `PROTOCOLS = {"A": set(), "A+": {"op", "hol"}, "A+W*": {"op", "hol", "wx_obs"}, "B": {"op", "hol", "prod"}}` (FR-29; no `A+W`).
  - `WX = ("기온", "풍속", "습도", "강수량_증분")`, `XCOLS = ("생산량",) + WX`
  - `season(month: int) -> int` — 0 winter (11–2), 1 spring/fall (3–5, 9–10), 2 summer (6–8) (FR-30).
  - `load_panel(include_copies=False, include_suspect=False, unseal=False) -> dict` — the panel dict of "Shared contracts".
  - `usable_peak(panel) -> np.ndarray` (257 bool) — `days.n_missing ≤ 4` and `Y[d]` has ≥ 1 finite value (FR-38).
  - `role_idx(panel, fold: str, role: str) -> np.ndarray` (int, ascending) — days whose `fold` column equals `role`.
  - `inner_idx(panel, fold: str) -> np.ndarray` (int, ascending) — `train`-role days whose date ≥ (last `train` date − 6 days) (ruling R10). f1 → 06-29…07-05, f2 → 07-13…07-19, f3 → 07-27…08-02, f4 → 08-10…08-16, test → 08-24…08-30.
  - `cal_flags(panel, protocol: str) -> tuple[np.ndarray, np.ndarray]` — `(op_eff, hol_eff)` int8: protocol `A` → all ones / all zeros; otherwise `(panel["op"], panel["hol"])` (FR-29).

**Algorithm (FR-27, FR-28, DC1, DC2, DC7, DC10, DC14):**
1. Read the 15-min CSV (`infer_schema_length=None`), sort by datetime, reshape each column to 257×96. Read the day table CSV, parse `date` with `%Y.%m.%d`.
2. `Y` = `전력` (NaN at missing points). Set `Y[d] = NaN` for days with `is_copy` (both kinds) unless `include_copies`; set `Y[d] = NaN` and `X["생산량"][d] = NaN` for `is_suspect` days unless `include_suspect`. Covariates of copy days are kept.
3. Unless `unseal`: for days whose `test` column == `"test"`, set `Y` and every `X` array to NaN.
4. `op`, `hol` from `is_operating`, `is_holiday`; `dtype` from `daytype` (wk 0, sat 1, sun 2); `dow` = `date.weekday()`; `month` = `date.month`.
5. `전기요금(계절)` is never loaded into the panel (DC7).

- [ ] **Step 1: Write the failing test** — create `tests/test_features.py`:

```python
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from gmst import features as ft
from gmst import preprocess as pp


@pytest.fixture(scope="module")
def panel():
    return ft.load_panel()


def di(p, d):
    return p["dates"].index(d)


def upto(p, d):
    return np.array([x <= d for x in p["dates"]])


def test_shapes(panel):
    assert len(panel["dates"]) == 257 and panel["dates"][0] == date(2021, 1, 1)
    assert panel["Y"].shape == (257, 96) and panel["Y"].dtype == np.float64
    assert set(panel["X"]) == {"생산량", "기온", "풍속", "습도", "강수량_증분"}
    assert all(v.shape == (257, 96) and v.dtype == np.float64 for v in panel["X"].values())
    assert panel["is_missing"].dtype == bool and int(panel["is_missing"].sum()) == 74
    assert panel["days"].height == 257 and panel["days"]["date"].dtype == pl.Date
    for k in ("op", "hol", "dtype", "dow", "month"):
        assert panel[k].shape == (257,)
    assert int(panel["op"].sum()) == 195 and int(panel["hol"].sum()) == 8
    assert np.bincount(panel["dtype"]).tolist() == [183, 37, 37]


def test_counts(panel):
    Y = panel["Y"]
    assert np.isfinite(Y).sum() == 11352
    assert np.isfinite(Y[upto(panel, date(2021, 7, 5))]).sum() == 6240
    m = upto(panel, date(2021, 8, 30))
    assert np.isfinite(Y[m]).sum() == 11256
    has = np.isfinite(Y[m]).any(1)
    assert has.sum() == 118 and (has & (panel["op"][m] == 1)).sum() == 93


@pytest.mark.parametrize("kw,n", [({"include_copies": True}, 23064), ({"include_suspect": True}, 11544),
                                  ({"include_copies": True, "include_suspect": True}, 23256)])
def test_ablation_counts(kw, n):
    assert np.isfinite(ft.load_panel(**kw)["Y"]).sum() == n


def test_sealed(panel):
    t = ft.role_idx(panel, "test", "test")
    assert len(t) == 14 and panel["dates"][t[0]] == date(2021, 9, 1)
    assert np.isnan(panel["Y"][t]).all()
    assert all(np.isnan(v[t]).all() for v in panel["X"].values())
    assert np.isfinite(ft.load_panel(unseal=True)["X"]["기온"][t]).all()


def test_copy_mask(panel):
    i = di(panel, date(2021, 3, 1))
    assert np.isnan(panel["Y"][i]).all()
    assert np.isfinite(panel["X"]["생산량"][i]).all() and np.isfinite(panel["X"]["기온"][i]).all()
    assert np.isfinite(ft.load_panel(include_copies=True)["Y"][i]).all()


def test_edited_mask(panel):
    for d in (date(2021, 1, 1), date(2021, 1, 9), date(2021, 1, 10), date(2021, 1, 16),
              date(2021, 3, 7), date(2021, 3, 21), date(2021, 3, 28)):
        assert np.isnan(panel["Y"][di(panel, d)]).all()


def test_suspect_mask(panel):
    for d in (date(2021, 7, 13), date(2021, 7, 15)):
        i = di(panel, d)
        assert np.isnan(panel["Y"][i]).all() and np.isnan(panel["X"]["생산량"][i]).all()
        assert np.isfinite(panel["X"]["기온"][i]).all()
    s = ft.load_panel(include_suspect=True)
    assert np.isfinite(s["Y"][di(s, date(2021, 7, 13))]).all()


def test_missing_points(panel):
    assert np.isnan(panel["Y"][panel["is_missing"]]).all()


def test_season_indicator():
    df = pl.read_csv(pp.OUT, infer_schema_length=None).with_columns(
        pl.col("datetime").str.slice(5, 2).cast(pl.Int32).alias("month"))
    per = df.group_by("month").agg(pl.col("전기요금(계절)").n_unique().alias("n"),
                                   pl.col("전기요금(계절)").first().alias("v"))
    assert per["n"].unique().to_list() == [1]
    pairs = {(ft.season(m), v) for m, v in per.select("month", "v").iter_rows()}
    assert len({s for s, _ in pairs}) == len({v for _, v in pairs}) == len(pairs)
    assert [ft.season(m) for m in (1, 3, 6, 9, 10, 11, 12)] == [0, 1, 2, 1, 1, 0, 0]


def test_protocols(panel):
    assert ft.PROTOCOLS == {"A": set(), "A+": {"op", "hol"}, "A+W*": {"op", "hol", "wx_obs"},
                            "B": {"op", "hol", "prod"}}
    op, hol = ft.cal_flags(panel, "A")
    assert (op == 1).all() and (hol == 0).all()
    op, hol = ft.cal_flags(panel, "A+")
    assert np.array_equal(op, panel["op"]) and np.array_equal(hol, panel["hol"])


def test_usable_peak(panel):
    u = ft.usable_peak(panel)
    assert [int(u[ft.role_idx(panel, f, "val")].sum()) for f in ("f1", "f2", "f3", "f4")] == [12, 13, 14, 12]


def test_role_and_inner_idx(panel):
    assert len(ft.role_idx(panel, "f1", "train")) == 186
    assert [panel["dates"][i] for i in ft.inner_idx(panel, "f1")] == [date(2021, 6, 29) + timedelta(days=k) for k in range(7)]
    assert [panel["dates"][i] for i in ft.inner_idx(panel, "f2")] == [date(2021, 7, d) for d in range(13, 20)]
    assert [panel["dates"][i] for i in ft.inner_idx(panel, "test")][-1] == date(2021, 8, 30)
    assert len(ft.inner_idx(panel, "f4")) == 7
```

- [ ] **Step 2: Run to verify it fails** — `uv run pytest tests/test_features.py -q` → `ModuleNotFoundError: No module named 'gmst.features'`.
- [ ] **Step 3: Implement** `gmst/features.py` (loader part only; Task 8 appends the B1 row builders).
- [ ] **Step 4: Run to verify it passes** — `uv run pytest tests/test_features.py -q` → `14 passed`.
- [ ] **Step 5: Full suite and commit**
```bash
git add gmst/features.py tests/test_features.py
git commit -m "feat: panel loader with copy/suspect/outage masking, protocols, seal, inner windows"
```

---

### Task 5: Metric functions and fold thresholds

**Files:**
- Create: `gmst/evaluate.py` (metrics part), `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `features.load_panel`, `features.role_idx`, `features.usable_peak`.
- Produces (module `gmst.evaluate`):
  - `TAUS = np.round(np.arange(1, 20) * 0.05, 2)`; `QCOLS = ["q05", "q10", …, "q95"]` (`f"q{round(t*100):02d}"`); `FOLDS_CV = ("f1", "f2", "f3", "f4")`; `CS = ("C50", "C75", "C90")`.
  - `mae(y, yhat) -> tuple[float, int]`, `rmse(y, yhat) -> tuple[float, int]` — over pairs where both are finite; n = pair count.
  - `crps(y, Q) -> tuple[float, int]` — `y` (n,), `Q` (n, 19) aligned with `TAUS`; rows with finite `y` and all-finite `Q`; value = 2 × mean over rows and τ of pinball `max(τ(y−q), (τ−1)(y−q))` (FR-37, A11).
  - `coverage(y, lo, hi) -> tuple[float, int]` — share of finite `y` with `lo ≤ y ≤ hi` (inclusive).
  - `brier(p, e) -> float` — mean `(p − e)²` over finite pairs.
  - `auc(p, e) -> float` — Mann–Whitney with ties counted 0.5; NaN when positives or negatives are absent (FR-41).
  - `prf(pred, true) -> dict` — keys `tp, fp, fn, precision, recall, f1`; precision NaN if no predicted positives; recall and f1 NaN if no true positives; `f1 = 2tp/(2tp+fp+fn)`.
  - `obs_max(Y, obs) -> float | np.ndarray` — max over slots where `obs` is True along the last axis; NaN if no observed slot. `Y` is (96,) or (N, 96) (FR-38).
  - `peak_mae(M_true, M_hat) -> tuple[float, int]`, `peak_hit(true_slot, pred_slot, k=2) -> tuple[float, int]` — over finite pairs; hit = `|true − pred| ≤ k`.
  - `thresholds(panel, train_idx) -> tuple[np.ndarray, int]` — days in `train_idx` with `op == 1` and `usable_peak`; daily max = `nanmax(Y[d])`; `C = np.quantile(maxima, [0.5, 0.75, 0.9])` (linear); returns `(C, n)` (FR-39, A1).

- [ ] **Step 1: Write the failing test** — create `tests/test_evaluate.py`:

```python
import numpy as np
import polars as pl
import pytest

from gmst import evaluate as ev
from gmst import features as ft


@pytest.fixture(scope="module")
def panel():
    return ft.load_panel()


def test_taus():
    assert len(ev.TAUS) == 19 and ev.TAUS[0] == pytest.approx(0.05) and ev.TAUS[-1] == pytest.approx(0.95)
    assert ev.QCOLS[:3] == ["q05", "q10", "q15"] and ev.QCOLS[9] == "q50" and ev.QCOLS[-1] == "q95"
    assert ev.FOLDS_CV == ("f1", "f2", "f3", "f4") and ev.CS == ("C50", "C75", "C90")


def test_mae_rmse_skip_nan():
    y = np.array([1.0, np.nan, 3.0])
    yh = np.array([2.0, 100.0, 1.0])
    assert ev.mae(y, yh) == (pytest.approx(1.5), 2)
    assert ev.rmse(y, yh) == (pytest.approx(np.sqrt(2.5)), 2)


def test_crps_is_twice_mean_pinball():
    assert ev.crps(np.array([0.0]), np.ones((1, 19))) == (pytest.approx(1.0), 1)
    assert ev.crps(np.array([0.0]), np.zeros((1, 19)))[0] == 0.0
    rng = np.random.default_rng(0)
    y = rng.normal(size=50)
    Q = np.sort(rng.normal(size=(50, 19)), axis=1)
    manual = np.mean([[max(t * (a - q), (t - 1) * (a - q)) for t, q in zip(ev.TAUS, row)] for a, row in zip(y, Q)])
    assert ev.crps(y, Q)[0] == pytest.approx(2 * manual)


def test_coverage():
    y = np.array([1.0, 5.0, np.nan, 3.0])
    assert ev.coverage(y, np.array([0, 0, 0, 3.0]), np.array([2, 4, 1, 4.0])) == (pytest.approx(2 / 3), 3)


def test_auc():
    assert ev.auc(np.array([0.1, 0.4, 0.35, 0.8]), np.array([0, 0, 1, 1])) == pytest.approx(0.75)
    assert ev.auc(np.array([0.5, 0.5, 0.5]), np.array([0, 1, 1])) == pytest.approx(0.5)
    assert ev.auc(np.array([0.9, 0.1]), np.array([0, 1])) == 0.0
    assert np.isnan(ev.auc(np.array([0.2, 0.3]), np.array([0, 0])))


def test_prf():
    r = ev.prf(np.array([1, 1, 0, 0, 1], bool), np.array([1, 0, 1, 0, 0], bool))
    assert (r["tp"], r["fp"], r["fn"]) == (1, 2, 1)
    assert r["precision"] == pytest.approx(1 / 3) and r["recall"] == pytest.approx(0.5) and r["f1"] == pytest.approx(0.4)
    r0 = ev.prf(np.array([0, 1], bool), np.array([0, 0], bool))
    assert np.isnan(r0["f1"]) and np.isnan(r0["recall"]) and r0["fp"] == 1


def test_brier():
    assert ev.brier(np.array([0.2, 0.9, np.nan]), np.array([0, 1, 1])) == pytest.approx((0.04 + 0.01) / 2)


def test_peak_same_slots():
    obs = np.ones(96, bool)
    obs[10] = False
    y = np.full(96, 50.0)
    y[10], y[40] = np.nan, 180.0
    paths = np.full((3, 96), 60.0)
    paths[:, 10] = 500.0
    paths[:, 40] = [170.0, 175.0, 190.0]
    assert ev.obs_max(y, obs) == 180.0
    assert np.median(ev.obs_max(paths, obs)) == 175.0
    assert np.isnan(ev.obs_max(np.full(96, np.nan), np.zeros(96, bool)))


def test_peak_mae_hit():
    assert ev.peak_mae(np.array([100.0, 200.0]), np.array([110.0, 190.0])) == (pytest.approx(10.0), 2)
    assert ev.peak_hit(np.array([10, 50]), np.array([12, 47]), k=2) == (pytest.approx(0.5), 2)


@pytest.mark.parametrize("fold,C,n", [
    ("f1", (181, 189.5, 198), 55), ("f2", (182, 193, 200.2), 65), ("f3", (184, 197.75, 210.1), 74),
    ("f4", (185, 198, 211), 81), ("test", (186.5, 198, 210.7), 92)])
def test_thresholds(panel, fold, C, n):
    got, cnt = ev.thresholds(panel, ft.role_idx(panel, fold, "train"))
    assert got == pytest.approx(C, abs=1e-9) and cnt == n


def test_event_counts(panel):
    u = ft.usable_peak(panel)
    out = []
    for f in ev.FOLDS_CV:
        C, _ = ev.thresholds(panel, ft.role_idx(panel, f, "train"))
        M = np.array([np.nanmax(panel["Y"][i]) for i in ft.role_idx(panel, f, "val") if u[i]])
        out.append([int((M > c).sum()) for c in C])
    assert out == [[8, 8, 4], [7, 7, 7], [6, 4, 2], [10, 2, 0]]
```

- [ ] **Step 2: Run to verify it fails** — `uv run pytest tests/test_evaluate.py -q` → `ModuleNotFoundError: No module named 'gmst.evaluate'`.
- [ ] **Step 3: Implement** the Task 5 functions in `gmst/evaluate.py` (numpy only).
- [ ] **Step 4: Run to verify it passes** — `uv run pytest tests/test_evaluate.py -q` → `15 passed`.
- [ ] **Step 5: Full suite and commit**
```bash
git add gmst/evaluate.py tests/test_evaluate.py
git commit -m "feat: metric functions (MAE/RMSE/CRPS/coverage/Brier/AUC/F1, same-slot peaks) and fold thresholds"
```

---

### Task 6: Naive baselines, rolling-origin loop, OOF tables, point metrics

**Files:**
- Create: `gmst/baselines.py`, `tests/test_baselines.py`
- Modify: `gmst/evaluate.py` (append)

**Interfaces:**
- Consumes: Task 4 panel helpers, Task 5 metrics.
- Produces:
  - `evaluate.point_pred(y: np.ndarray, C: np.ndarray) -> dict` — FR-48 prediction dict for point baselines: `y_mean = y_median = y`, `q = None`, `paths = None`, `M_hat_median = M_hat_mean = nanmax(y)`, `peak_time_mode = int(nanargmax(y))`, `risk_raw = (M_hat_median > C).astype(float)`.
  - `evaluate.SLOT_COLS = ["fold", "variant", "model", "date", "datetime", "y_true", "is_missing", "y_mean", "y_median", *QCOLS, "state_filt", "state_smooth"]`
  - `evaluate.DAY_COLS = ["fold", "variant", "model", "date", "usable_peak", "n_obs", "M_true", "M_hat_median", "M_hat_mean", "peak_slot_true", "peak_time_mode", "C50", "C75", "C90", "risk_raw_C50", "risk_raw_C75", "risk_raw_C90", "risk_platt_C50", "risk_platt_C75", "risk_platt_C90", "risk_iso_C50", "risk_iso_C75", "risk_iso_C90", "event_C50", "event_C75", "event_C90"]` (FR-43)
  - `evaluate.rolling_origin(model, panel, folds=FOLDS_CV, variant="main", states=None) -> tuple[pl.DataFrame, pl.DataFrame, dict, pl.DataFrame]` = `(slots, days, states, inner_days)`
  - `evaluate.point_metrics(slots, days, panel) -> pl.DataFrame` with columns `["model", "variant", "fold", "stratum", "metric", "value", "n"]` (FR-44)
  - `baselines.b0p_ref(panel, d, limit=28) -> tuple[int | None, bool]` — (reference day index, fallback_used)
  - `baselines.b0p_model(limit=28) -> dict` (name `B0p`), `baselines.b0_model() -> dict` (name `B0`)
  - `baselines.lag7_skip_table(panel) -> pl.DataFrame` columns `fold, mae, n` (rows f1…f4)

**Algorithm:**
- `b0p_ref` (FR-45, ruling R4): candidates `j < d` with `np.isfinite(Y[j]).all()` (this covers loader-masked days, sealed days and `n_missing > 0`); first pass `j` from `d−1` down to `max(d−limit, 0)` with `op[j] == op[d]` and `dtype[j] == dtype[d]` → `(j, False)`; `limit=None` means no window. Otherwise second pass from `d−1` down to 0 with `op[j] == op[d]` only → `(j, True)`; none → `(None, True)`. Never a future day, never a negative index.
- `b0p_model`: `fit(panel, fold, C) -> {"C": C}`; `predict` → `point_pred(Y[ref], C)` (if `ref is None`, all-NaN y).
- `b0_model` (FR-46): `y = Y[d−7]` (all-NaN if `d < 7`), NaN slots replaced by the B0′ values for day `d`; → `point_pred`.
- `lag7_skip_table` (FR-47): per fold, over val days, points where `Y[d]` and `Y[d−7]` are both finite; MAE and n.
- `rolling_origin` (FR-43): for each fold `f`: `tr = role_idx(panel, f, "train")`; `C, _ = thresholds(panel, tr)`; `state = states[f] if states else model["fit"](panel, f, C)`; validation days = `role_idx(panel, f, "test" if f == "test" else "val")` in date order; `pred = model["predict"](state, panel, d)`; parameters fixed within the fold, history grows by using the panel as-is.
  - Slot rows (96 per day): `date` `%Y.%m.%d`, `datetime` `%Y.%m.%d %H:%M:%S`, `y_true = Y[d, q]` (NaN when masked), `is_missing = panel["is_missing"][d, q]`, `y_mean`, `y_median`, `q05…q95` (NaN when `q is None`), `state_filt`/`state_smooth` (Int64, null unless `"posthoc"` in model; then `model["posthoc"](state, panel, d)` returns two int arrays of 1-based states).
  - Day row: `obs = isfinite(Y[d])`; `usable_peak = usable_peak(panel)[d]`; `n_obs = obs.sum()`; `M_true = obs_max(Y[d], obs)`; `peak_slot_true = nanargmax(Y[d])` (Int64, null if no obs); evaluation `M_hat_*` (FR-38): if `pred["paths"]` is not None → `median`/`mean` of `obs_max(paths, obs)` when `n_obs > 0` else of the 96-slot path maxima; elif `pred["q"] is None` (point baseline) → `obs_max(y_median, obs)` for both (or `nanmax` if no obs); else (B1) the predicted `M_hat_median`/`M_hat_mean` unchanged. `risk_raw_*` = `pred["risk_raw"]` as issued (not recomputed on observed slots — **[plan pin]**, matters only for 09-08); `risk_platt_*`, `risk_iso_*` = NaN (filled in Task 9); `event_Cj = float(M_true > C_j)` if `usable_peak` else NaN; `C50/C75/C90` = fold thresholds.
  - Inner rows: if `"inner_state" in state`, the same day rows (no slot rows) for `inner_idx(panel, f)` predicted with `model["predict"](state["inner_state"], panel, d)`; else none. `inner_days` has `DAY_COLS`.
  - Returns `slots` (SLOT_COLS), `days` (DAY_COLS), `states` ({fold: state}), `inner_days` (DAY_COLS, possibly empty).
- `point_metrics`: for each (model, variant, fold) present plus `pooled` (all rows with fold in `FOLDS_CV`, emitted when at least one CV fold is present) and each stratum `all` / `op` / `nonop` (true `panel["op"]` of the row's date): slot metrics `mae` (y_median), `rmse` (y_mean), `crps`, `cov50` (q25,q75), `cov80` (q10,q90), `cov90` (q05,q95) with n = points used (NaN value, n 0 when q is absent); day metrics over `usable_peak` rows: `peak_mae`, `peak_hit2`, `brier_raw_C50/75/90`, `brier_mean_raw` (mean of the three), `auc_C50/75/90` (on `risk_raw`), `n_events_C50/75/90` (value = event count, n = usable days). `value` Float64, `n` Int64.

- [ ] **Step 1: Write the failing test** — create `tests/test_baselines.py`:

```python
from datetime import date

import numpy as np
import polars as pl
import pytest

from gmst import baselines as bl
from gmst import evaluate as ev
from gmst import features as ft

FOLDS = ("f1", "f2", "f3", "f4")


@pytest.fixture(scope="module")
def panel():
    return ft.load_panel()


@pytest.fixture(scope="module")
def b0p(panel):
    return ev.rolling_origin(bl.b0p_model(), panel)


def fold_mae(slots, fold):
    s = slots.filter(pl.col("fold") == fold)
    return ev.mae(s["y_true"].to_numpy(), s["y_median"].to_numpy())


def test_lag7_skip_table(panel):
    t = bl.lag7_skip_table(panel)
    assert t["fold"].to_list() == list(FOLDS)
    assert [round(v, 2) for v in t["mae"].to_list()] == [6.23, 25.77, 65.34, 9.84]
    assert t["n"].to_list() == [960, 1152, 1248, 1272]


def test_b0p_oof(b0p):
    slots = b0p[0]
    got = [fold_mae(slots, f) for f in FOLDS]
    assert [round(m, 2) for m, _ in got] == [14.85, 10.38, 14.49, 15.17]
    assert [n for _, n in got] == [1152, 1248, 1344, 1272]


def test_b0p_fallback_only_two_days(panel):
    fb = {}
    for f in FOLDS:
        for d in ft.role_idx(panel, f, "val"):
            ref, used = bl.b0p_ref(panel, int(d))
            assert ref < d and np.isfinite(panel["Y"][ref]).all()
            assert panel["days"]["n_missing"][ref] == 0
            if used:
                fb[panel["dates"][d]] = panel["dates"][ref]
    assert fb == {date(2021, 7, 31): date(2021, 7, 25), date(2021, 8, 2): date(2021, 8, 1)}


def test_b0p_no_limit_sensitivity(panel):
    slots = ev.rolling_origin(bl.b0p_model(limit=None), panel, folds=("f2",))[0]
    assert round(fold_mae(slots, "f2")[0], 2) == 12.06
    ref, _ = bl.b0p_ref(panel, panel["dates"].index(date(2021, 7, 31)), limit=None)
    assert panel["dates"][ref] == date(2021, 1, 2)


def test_b0_predicts_every_slot(panel):
    slots = ev.rolling_origin(bl.b0_model(), panel)[0]
    assert slots.height == 5376 and np.isfinite(slots["y_median"].to_numpy()).all()


def test_oof_schema(b0p):
    slots, days, states, inner = b0p
    assert slots.columns == ev.SLOT_COLS and days.columns == ev.DAY_COLS and inner.columns == ev.DAY_COLS
    assert slots.height == 5376 and days.height == 56 and inner.height == 0
    assert set(slots["model"].unique()) == {"B0p"} and set(days["variant"].unique()) == {"main"}
    assert days.filter(pl.col("usable_peak")).height == 51
    assert set(states) == set(FOLDS)
    d0 = days.filter(pl.col("fold") == "f1").row(0, named=True)
    assert d0["date"] == "2021.07.07" and (d0["C50"], d0["C75"], d0["C90"]) == pytest.approx((181.0, 189.5, 198.0))
    assert slots["datetime"][0] == "2021.07.07 00:00:00"
    risk = days.select("risk_raw_C50", "risk_raw_C75", "risk_raw_C90").to_numpy()
    assert set(np.unique(risk)) <= {0.0, 1.0}


def test_point_metrics(b0p, panel):
    slots, days, _, _ = b0p
    m = ev.point_metrics(slots, days, panel)
    assert m.columns == ["model", "variant", "fold", "stratum", "metric", "value", "n"]
    r = m.filter((pl.col("fold") == "f1") & (pl.col("stratum") == "all") & (pl.col("metric") == "mae")).row(0, named=True)
    assert round(r["value"], 2) == 14.85 and r["n"] == 1152
    assert set(m["fold"].unique()) == {"f1", "f2", "f3", "f4", "pooled"}
    assert set(m["stratum"].unique()) == {"all", "op", "nonop"}
    ne = m.filter((pl.col("stratum") == "all") & (pl.col("metric") == "n_events_C90"))
    assert dict(ne.select("fold", "value").iter_rows()) == {"f1": 4, "f2": 7, "f3": 2, "f4": 0, "pooled": 13}
    assert np.isnan(m.filter(pl.col("metric") == "crps")["value"].to_numpy()).all()
    auc4 = m.filter((pl.col("fold") == "f4") & (pl.col("stratum") == "all") & (pl.col("metric") == "auc_C90"))
    assert np.isnan(auc4["value"][0])
```

- [ ] **Step 2: Run to verify it fails** — `uv run pytest tests/test_baselines.py -q` → `ModuleNotFoundError: No module named 'gmst.baselines'`.
- [ ] **Step 3: Implement** `gmst/baselines.py` (B0, B0′, lag-7 table) and append `point_pred`, `SLOT_COLS`, `DAY_COLS`, `rolling_origin`, `point_metrics` to `gmst/evaluate.py`.
- [ ] **Step 4: Run to verify it passes** — `uv run pytest tests/test_baselines.py -q` → `7 passed`. These reproduce v2 §4.3 / 부록 B (G3).
- [ ] **Step 5: Full suite and commit**
```bash
git add gmst/baselines.py gmst/evaluate.py tests/test_baselines.py
git commit -m "feat: B0/B0' baselines, rolling-origin OOF loop with inner-day rows, point metrics (reproduces v2 §4.3)"
```

---

### Task 7: Penalized cyclic-RW2 backbone, BB model, (τ, h) selection on inner windows

**Files:**
- Create: `gmst/backbone.py`, `tests/test_backbone.py`

**Interfaces:**
- Consumes: panel helpers (`role_idx`, `inner_idx`, `cal_flags`), `evaluate.point_pred`, `evaluate.rolling_origin`, `evaluate.FOLDS_CV`.
- Produces (module `gmst.backbone`):
  - `C2: np.ndarray` (96×96) — row q has `+1` at `(q−1) mod 96`, `−2` at `q`, `+1` at `(q+1) mod 96`.
  - `TAU_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0)`, `H_GRID = (30, 60, 120)`
  - `fit_backbone(Y, op, daytype, train_idx, tau, half_life=60, min_days=5) -> tuple[np.ndarray, list[tuple[int, int]]]` — profiles (2, 3, 96) indexed `[op, dtype]`, and the sorted list of fallback classes.
  - `predict_backbone(profiles, op, daytype) -> np.ndarray` — `profiles[op, daytype]` → (n, 96).
  - `backbone_asof(Y, op, daytype, d, tau, half_life, min_days=5) -> np.ndarray` (96,)
  - `asof_matrix(panel, day_idx, tau, half_life, protocol="A+") -> np.ndarray` (257, 96), rows `day_idx` filled, others NaN.
  - `fold_backbone(panel, train_idx, tau, half_life, protocol="A+") -> np.ndarray` (257, 96) — m for every day from one fit on `train_idx`.
  - `bb_model(tau, half_life, protocol="A+") -> dict` (name `BB`)
  - `select_tau(panel, folds=FOLDS_CV) -> tuple[float, int, pl.DataFrame]` — `(tau, h, table)` with table columns `tau, half_life, mae_pooled` (15 rows, grid order).

**Algorithm (v2 §6, FR-58…FR-62, A5, A6, A23, A32, ruling R10):**
- `fit_backbone`: weights `w_d = 2 ** (−(last − d) / half_life)` with `last = max(train_idx)` (`half_life=None` → 1). Usable class days = train days with ≥ 1 finite Y. For op `o` the parent system pools all its dtypes: `W_p[q] = Σ_d w_d·1[finite]`, `b_p[q] = Σ_d w_d·Y[d,q]` (NaN → 0), parent profile = `np.linalg.solve(np.diag(W_p) + tau * C2.T @ C2, b_p)` (NaN profile if the parent has no data). Each class `(o, t)` with fewer than `min_days` usable days gets the parent profile and is appended to the fallback list; otherwise it is solved with its own `W_c, b_c`. No sum-to-zero constraint, no trend term, no posterior sd (FR-59, FR-62).
- `backbone_asof(Y, op, daytype, d, …)` is defined as `fit_backbone(Y, op, daytype, np.arange(d − 1), …)[0][op[d], daytype[d]]` (data through `d−2`, A32) and all-NaN for `d < 2`. It must call `fit_backbone` exactly this way so the test can compare bitwise.
- `asof_matrix` uses `op_eff, _ = cal_flags(panel, protocol)` and `panel["dtype"]`; `fold_backbone` likewise.
- `bb_model`: `fit(panel, fold, C) -> {"m": fold_backbone(panel, role_idx(panel, fold, "train"), tau, h, protocol), "C": C}`; `predict → point_pred(state["m"][d], C)` (FR-51).
- `select_tau` (FR-61 with R10): for every `(tau, h)` in grid order, for every fold in `folds`: `tr = role_idx(panel, f, "train")`, `inn = inner_idx(panel, f)`, fit on `setdiff(tr, inn)` (true op flags), predict the inner days, accumulate absolute errors and counts over finite `Y`; `mae_pooled = Σ|err| / Σn`. Choose the minimum (ties → first in grid order). If the choice is on a grid edge (`tau ∈ {0.1, 1000}` or `h ∈ {30, 120}`) print `backbone grid edge: tau=<tau> h=<h>`. The same `(tau, h)` is used for every fold refit, the test fit and the B1 as-of feature (**[plan pin]**: one global pair).

- [ ] **Step 1: Write the failing test** — create `tests/test_backbone.py`:

```python
from datetime import date

import numpy as np
import polars as pl
import pytest

from gmst import backbone as bb
from gmst import evaluate as ev
from gmst import features as ft


@pytest.fixture(scope="module")
def panel():
    return ft.load_panel()


def window(p, a, b):
    return np.array([i for i, d in enumerate(p["dates"]) if a <= d <= b])


def cell_stats(p, idx):
    prof, _ = bb.fit_backbone(p["Y"], p["op"], p["dtype"], idx, tau=0.0, half_life=None, min_days=1)
    m = bb.predict_backbone(prof, p["op"][idx], p["dtype"][idx])
    y = p["Y"][idx]
    r = y - m
    f = np.isfinite(y)
    r2 = 1 - np.sum(r[f] ** 2) / np.sum((y[f] - y[f].mean()) ** 2)
    opm = np.repeat(p["op"][idx][:, None], 96, 1) == 1
    sd1, sd0 = np.std(r[f & opm], ddof=1), np.std(r[f & ~opm], ddof=1)
    flat = r.reshape(-1)
    c = flat[np.isfinite(flat)]
    lag1 = np.corrcoef(c[1:], c[:-1])[0, 1]   # [plan pin] drop NaNs, pair consecutive observed residuals
    return r2, int(f.sum()), sd1, sd0, lag1, r


def test_c2():
    assert bb.C2.shape == (96, 96)
    assert np.allclose(bb.C2 @ np.ones(96), 0)
    r = bb.C2 @ np.arange(96.0)
    assert r[0] == 96 and r[95] == -96 and np.allclose(r[1:95], 0)


def test_cell_mean_julaug(panel):
    r2, n, sd1, sd0, lag1, _ = cell_stats(panel, window(panel, date(2021, 7, 1), date(2021, 8, 31)))
    assert n == 5592 and r2 == pytest.approx(0.923, abs=5e-4)
    assert sd1 == pytest.approx(19.95, abs=0.02) and sd0 == pytest.approx(1.55, abs=0.02)
    assert lag1 == pytest.approx(0.909, abs=0.002)


def test_cell_mean_le0831(panel):
    idx = window(panel, date(2021, 1, 1), date(2021, 8, 31))
    _, _, sd1, sd0, lag1, r = cell_stats(panel, idx)
    assert sd1 == pytest.approx(32.36, abs=0.02) and sd0 == pytest.approx(7.31, abs=0.02)
    assert lag1 == pytest.approx(0.961, abs=0.002)
    keep = np.array([panel["dates"][i] != date(2021, 1, 2) and panel["op"][i] == 0 for i in idx])
    rr = r[keep]
    assert np.std(rr[np.isfinite(rr)], ddof=1) == pytest.approx(4.47, abs=0.02)   # 01-02 residuals dropped, means kept


def test_large_tau_constant(panel):
    prof, _ = bb.fit_backbone(panel["Y"], panel["op"], panel["dtype"], ft.role_idx(panel, "f4", "train"), tau=1e8)
    assert all(np.std(prof[o, t]) < 1e-3 for o in (0, 1) for t in (0, 1, 2))


def test_fallback_f1(panel):
    _, fb = bb.fit_backbone(panel["Y"], panel["op"], panel["dtype"], ft.role_idx(panel, "f1", "train"), tau=10.0)
    assert (1, 2) in fb and (0, 1) in fb


def test_invariance_to_validation_window(panel):
    tr = ft.role_idx(panel, "f2", "train")
    a, _ = bb.fit_backbone(panel["Y"], panel["op"], panel["dtype"], tr, tau=10.0, half_life=60)
    Y2 = panel["Y"].copy()
    Y2[ft.role_idx(panel, "f2", "val")] = 999.0
    b, _ = bb.fit_backbone(Y2, panel["op"], panel["dtype"], tr, tau=10.0, half_life=60)
    assert np.array_equal(a, b)


def test_backbone_asof_uses_d_minus_2(panel):
    d = panel["dates"].index(date(2021, 7, 21))
    got = bb.backbone_asof(panel["Y"], panel["op"], panel["dtype"], d, tau=10.0, half_life=60)
    prof, _ = bb.fit_backbone(panel["Y"], panel["op"], panel["dtype"], np.arange(d - 1), tau=10.0, half_life=60)
    assert np.array_equal(got, prof[panel["op"][d], panel["dtype"][d]])
    Y2 = panel["Y"].copy()
    Y2[d - 1:] = 999.0
    assert np.array_equal(got, bb.backbone_asof(Y2, panel["op"], panel["dtype"], d, tau=10.0, half_life=60))


def test_select_tau(panel):
    tau, h, table = bb.select_tau(panel)
    assert table.columns == ["tau", "half_life", "mae_pooled"] and table.height == 15
    best = table.sort("mae_pooled").row(0, named=True)
    assert (tau, h) == (best["tau"], best["half_life"])
    assert sorted(table["tau"].unique().to_list()) == [0.1, 1.0, 10.0, 100.0, 1000.0]


def test_select_tau_never_reads_f4_validation(panel):
    p2 = {**panel, "Y": panel["Y"].copy()}
    p2["Y"][ft.role_idx(panel, "f4", "val")] = 999.0
    assert bb.select_tau(panel)[2].equals(bb.select_tau(p2)[2])


def test_bb_model_oof(panel):
    slots = ev.rolling_origin(bb.bb_model(10.0, 60), panel, folds=("f4",))[0]
    assert slots.height == 1344 and np.isfinite(slots["y_median"].to_numpy()).all()
    assert np.array_equal(slots["y_mean"].to_numpy(), slots["y_median"].to_numpy())
```

- [ ] **Step 2: Run to verify it fails** — `uv run pytest tests/test_backbone.py -q` → `ModuleNotFoundError: No module named 'gmst.backbone'`.
- [ ] **Step 3: Implement** `gmst/backbone.py`.
- [ ] **Step 4: Run to verify it passes** — `uv run pytest tests/test_backbone.py -q` → `10 passed`.
- [ ] **Step 5: Full suite and commit**
```bash
git add gmst/backbone.py tests/test_backbone.py
git commit -m "feat: closed-form cyclic RW2 backbone, BB model, inner-window (tau, h) selection, as-of backbone"
```

---

### Task 8: B1 LightGBM strong baseline and the oracle-weather ablation (A+W*)

**Files:**
- Modify: `gmst/features.py` (append row builders), `gmst/baselines.py` (append B1), `tests/test_features.py` (append), `tests/test_baselines.py` (append)

**Interfaces:**
- Consumes: Task 4 panel, Task 7 `asof_matrix`, Task 5/6 `thresholds`, `TAUS`, `rolling_origin`.
- Produces:
  - `features.SLOT_FEATURES` (A+ list, 30 names, order below), `features.PROTO_SLOT = {"B": ["prod_h", "prod_on"], "A+W*": ["ob_temp", "ob_rain", "ob_wind", "ob_hum"]}`, `features.DAY_FEATURES` (13 names), `features.PROTO_DAY = {"B": ["prod_sum", "prod_hours"], "A+W*": ["ob_temp_mean", "ob_rain_sum", "ob_wind_mean", "ob_hum_mean"]}`.
  - `features.lgbm_rows(panel, day_idx, protocol, backbone) -> tuple[np.ndarray, np.ndarray, list[str]]` — `X` float64 (96·len(day_idx), F), `y` (96·len,), names; rows day-major then q ascending; `backbone` is a (257, 96) array whose row `d` is the as-of backbone of day `d`.
  - `features.day_rows(panel, day_idx, protocol, backbone) -> tuple[np.ndarray, list[str]]` — one row per day.
  - `baselines.LGB_PARAMS = {"seed": 0, "deterministic": True, "force_col_wise": True, "verbose": -1}` with the exact marker on that line: `# ponytail: 기본 하이퍼파라미터, B1이 f1–f4에서 B0′를 못 이기면 튜닝 (FR-49)`.
  - `baselines.fit_b1(panel, train_idx, protocol, backbone, C, rounds=100) -> dict` — `backbone = (tau, half_life)`; returns `{"tau", "h", "protocol", "C", "rounds", "names", "n_rows", "mean", "q" (19 boosters), "Mq" (19 boosters), "clf" (3 boosters or None)}`.
  - `baselines.predict_b1(models, panel, d) -> dict` — prediction dict + `"Mq"` (19 sorted day quantiles).
  - `baselines.b1_model(tau, half_life, protocol="A+", rounds=100) -> dict` (name `B1`); its `fit(panel, fold, C)` returns the full-train state with `"inner_state"` = `fit_b1` on `train \ inner_idx(fold)`.

**Feature definitions (FR-31, FR-32, v2 §15.4, A32).** For target day `d`, slot `q`, `op_eff, hol_eff = cal_flags(panel, protocol)`; any index < 0 gives NaN:
- `q`, `q_mod4 = q % 4`, `sin1, cos1, sin2, cos2, sin3, cos3` = `sin/cos(2π r q / 96)`, r = 1, 2, 3; `dow`, `daytype` (0/1/2), `month`, `season` (`features.season`), `hol` (= `hol_eff[d]`), `op` (= `op_eff[d]`).
- `y_lag1, y_lag2, y_lag7` = `Y[d−1, q], Y[d−2, q], Y[d−7, q]`.
- `prev_last4_mean` = nanmean of `Y[d−1, 92:96]`; `prev_mean, prev_max, prev_min` = nan-aggregates of `Y[d−1]` (NaN when all NaN).
- `recent_same_mean` = per-slot nanmean over `j ∈ [d−7, d−1]` with `dtype[j] == dtype[d]`, `op_eff[j] == op_eff[d]` and ≥ 1 finite `Y[j]` (NaN if none).
- `op_lag1, op_lag7` = `op_eff[d−1], op_eff[d−7]`; `backbone` = `backbone[d, q]`.
- `prev_temp, prev_hum, prev_wind` = mean of `X[기온|습도|풍속][d−1]` over 96 slots (NaN if any NaN); `prev_rain` = `sum(X["강수량_증분"][d−1]) / 4`; `prev_prod` = `sum(X["생산량"][d−1]) / 4` (NaN if any NaN).
- Protocol `B`: `prod_h = X["생산량"][d, q]`, `prod_on = float(prod_h > 0)`. Protocol `A+W*`: `ob_temp, ob_rain, ob_wind, ob_hum` = `X[기온|강수량_증분|풍속|습도][d, q]`. `A` and `A+` share the base list (A has constant `op=1`, `hol=0`).
- Day features: `prev_mean, prev_max, prev_min, y_lag7_mean` (nanmean `Y[d−7]`), `y_lag7_max`, `recent_same_max` (nanmax over q of `recent_same_mean`), `backbone_max` (max of `backbone[d]`), `dow, daytype, month, hol, op, op_lag1`; B adds `prod_sum = sum(X["생산량"][d]) / 4`, `prod_hours = (X["생산량"][d] > 0).sum() / 4`; A+W* adds `ob_temp_mean, ob_rain_sum (= sum/4), ob_wind_mean, ob_hum_mean` of day `d`.

**B1 algorithm (FR-49, FR-50, A14):**
1. `bbm = backbone.asof_matrix(panel, train_idx, tau, h, protocol)`; rows from `lgbm_rows`; keep rows with finite `y` (`n_rows`).
2. `mean` = `lgb.train({**LGB_PARAMS, "objective": "regression"}, lgb.Dataset(X, y), rounds)`; 19 quantile boosters `{"objective": "quantile", "alpha": τ}`.
3. Day model rows = `train_idx` days with `usable_peak`; label `M = nanmax(Y[d])` (observed slots). 19 quantile boosters on `M` (`Mq`). For each threshold `j`: if both classes occur in `M > C_j`, a binary booster (`"objective": "binary"`); else `None` and print one line `B1: C<50|75|90> single-class in training; risk = 1 - F_M(C)`.
4. `predict_b1`: as-of backbone row for `d` only; `y_mean` from `mean`; `q = np.sort(19 predictions, axis=0)`; `y_median = q[9]`; `Mq = np.sort(day quantile predictions)`; `M_hat_median = Mq[9]`, `M_hat_mean = Mq.mean()`; `peak_time_mode = int(np.argmax(y_median))`; `risk_raw[j]` = booster probability, or `1 − np.interp(C_j, Mq, TAUS, left=0.0, right=1.0)` when its classifier is `None`; `paths = None`.
5. The oracle-weather ablation is `b1_model(tau, h, protocol="A+W*")` run as variant `AWstar` (FR-81, ruling R9: a heuristic upper bound). It is never a B4-family candidate, never in the final fit and never in `test_predictions.csv` (checked in Task 18).

- [ ] **Step 1: Write the failing tests** — append to `tests/test_features.py`:

```python
A_PLUS = ["q", "q_mod4", "sin1", "cos1", "sin2", "cos2", "sin3", "cos3", "dow", "daytype", "month", "season",
          "hol", "op", "y_lag1", "y_lag2", "y_lag7", "prev_last4_mean", "prev_mean", "prev_max", "prev_min",
          "recent_same_mean", "op_lag1", "op_lag7", "backbone", "prev_temp", "prev_hum", "prev_wind", "prev_rain",
          "prev_prod"]
DAY_A_PLUS = ["prev_mean", "prev_max", "prev_min", "y_lag7_mean", "y_lag7_max", "recent_same_max", "backbone_max",
              "dow", "daytype", "month", "hol", "op", "op_lag1"]
ZB = np.zeros((257, 96))


def rows(p, d, protocol="A+"):
    return ft.lgbm_rows(p, np.array([p["dates"].index(d)]), protocol, ZB)


def test_feature_names(panel):
    d = date(2021, 8, 18)
    assert ft.SLOT_FEATURES == A_PLUS and ft.DAY_FEATURES == DAY_A_PLUS
    assert rows(panel, d)[2] == A_PLUS and rows(panel, d, "A")[2] == A_PLUS
    assert rows(panel, d, "B")[2] == A_PLUS + ["prod_h", "prod_on"]
    assert rows(panel, d, "A+W*")[2] == A_PLUS + ["ob_temp", "ob_rain", "ob_wind", "ob_hum"]


def test_lag_references_masked_days(panel):
    for d in (date(2021, 7, 20), date(2021, 7, 22), date(2021, 1, 8)):
        X, _, names = rows(panel, d)
        assert np.isnan(X[:, names.index("y_lag7")]).all()
    X, _, names = rows(panel, date(2021, 1, 2))          # d-1 = 01-01, an edited copy
    for c in ("y_lag1", "prev_mean", "prev_max", "prev_min", "prev_last4_mean"):
        assert np.isnan(X[:, names.index(c)]).all()


def test_rows_values(panel):
    d = date(2021, 8, 18)
    i = panel["dates"].index(d)
    X, y, names = rows(panel, d)
    col = lambda c: X[:, names.index(c)]
    assert X.shape == (96, 30) and np.array_equal(y, panel["Y"][i], equal_nan=True)
    assert np.array_equal(col("q"), np.arange(96)) and np.array_equal(col("q_mod4"), np.arange(96) % 4)
    assert np.array_equal(col("y_lag1"), panel["Y"][i - 1], equal_nan=True)
    assert col("prev_max")[0] == np.nanmax(panel["Y"][i - 1])
    assert col("prev_prod")[0] == pytest.approx(panel["X"]["생산량"][i - 1].sum() / 4)
    assert col("prev_rain")[0] == pytest.approx(panel["X"]["강수량_증분"][i - 1].sum() / 4)
    assert col("op")[0] == panel["op"][i] and col("month")[0] == 8 and col("season")[0] == 2
    assert col("sin1")[24] == pytest.approx(1.0)


def test_aplus_ignores_same_day_exogenous(panel):
    i = panel["dates"].index(date(2021, 8, 18))
    base = ft.lgbm_rows(panel, np.array([i]), "A+", ZB)[0]
    p2 = {**panel, "X": {k: v.copy() for k, v in panel["X"].items()}}
    for k in p2["X"]:
        p2["X"][k][i] += 7.0
    assert np.array_equal(ft.lgbm_rows(p2, np.array([i]), "A+", ZB)[0], base, equal_nan=True)
    b0 = ft.lgbm_rows(panel, np.array([i]), "B", ZB)[0]
    b1 = ft.lgbm_rows(p2, np.array([i]), "B", ZB)[0]
    assert not np.array_equal(b0, b1, equal_nan=True)


def test_day_rows(panel):
    i = panel["dates"].index(date(2021, 8, 18))
    X, names = ft.day_rows(panel, np.array([i]), "A+", ZB)
    assert names == DAY_A_PLUS and X.shape == (1, 13)
    assert ft.day_rows(panel, np.array([i]), "B", ZB)[1] == DAY_A_PLUS + ["prod_sum", "prod_hours"]
    assert ft.day_rows(panel, np.array([i]), "A+W*", ZB)[1] == DAY_A_PLUS + [
        "ob_temp_mean", "ob_rain_sum", "ob_wind_mean", "ob_hum_mean"]
```

Append to `tests/test_baselines.py`:

```python
from pathlib import Path


@pytest.fixture(scope="module")
def b1(panel):
    C, _ = ev.thresholds(panel, ft.role_idx(panel, "f1", "train"))
    model = bl.b1_model(10.0, 60)
    return model, model["fit"](panel, "f1", C)


def test_b1_train_rows(b1):
    _, state = b1
    assert state["n_rows"] == 6240 and state["inner_state"]["n_rows"] == 5760
    assert len(state["q"]) == 19 and len(state["Mq"]) == 19 and len(state["clf"]) == 3


def test_b1_final_rows(panel):
    _, y, _ = ft.lgbm_rows(panel, ft.role_idx(panel, "test", "train"), "A+", np.zeros((257, 96)))
    assert np.isfinite(y).sum() == 11256


def test_b1_prediction_contract(panel, b1):
    model, state = b1
    p = model["predict"](state, panel, panel["dates"].index(date(2021, 7, 7)))
    assert p["q"].shape == (19, 96) and (np.diff(p["q"], axis=0) >= 0).all()
    assert np.array_equal(p["y_median"], p["q"][9]) and p["y_mean"].shape == (96,) and p["paths"] is None
    assert p["peak_time_mode"] == int(np.argmax(p["y_median"]))
    assert p["risk_raw"].shape == (3,) and ((p["risk_raw"] >= 0) & (p["risk_raw"] <= 1)).all()
    assert p["M_hat_median"] == p["Mq"][9] and p["M_hat_mean"] == pytest.approx(np.mean(p["Mq"]))


def test_b1_no_lookahead(panel, b1):
    model, state = b1
    d = panel["dates"].index(date(2021, 7, 7))
    a = model["predict"](state, panel, d)
    rng = np.random.default_rng(1)
    p = {**panel, "Y": panel["Y"].copy(), "X": {k: v.copy() for k, v in panel["X"].items()}}
    p["Y"][d:] = rng.uniform(0, 250, p["Y"][d:].shape)
    for v in p["X"].values():
        v[d:] = rng.uniform(0, 30, v[d:].shape)
    b = model["predict"](state, p, d)
    for k in ("y_mean", "y_median", "q", "risk_raw", "Mq"):
        assert np.array_equal(a[k], b[k]), k


def test_awstar_uses_same_day_weather(panel):
    C, _ = ev.thresholds(panel, ft.role_idx(panel, "f1", "train"))
    m = bl.b1_model(10.0, 60, protocol="A+W*")
    st = m["fit"](panel, "f1", C)
    d = panel["dates"].index(date(2021, 7, 7))
    p = {**panel, "X": {k: v.copy() for k, v in panel["X"].items()}}
    for k in ("기온", "풍속", "습도", "강수량_증분"):
        p["X"][k][d] = p["X"][k][d] + 25.0
    assert not np.array_equal(m["predict"](st, panel, d)["y_mean"], m["predict"](st, p, d)["y_mean"])


def test_lgbm_defaults_marker():
    assert bl.LGB_PARAMS == {"seed": 0, "deterministic": True, "force_col_wise": True, "verbose": -1}
    src = Path(bl.__file__).read_text(encoding="utf-8")
    assert "# " + "ponytail: 기본 하이퍼파라미터, B1이 f1–f4에서 B0′를 못 이기면 튜닝" in src
```

- [ ] **Step 2: Run to verify they fail** — `uv run pytest tests/test_features.py tests/test_baselines.py -q` → failures with `AttributeError: module 'gmst.features' has no attribute 'SLOT_FEATURES'` and `… 'gmst.baselines' has no attribute 'b1_model'`.
- [ ] **Step 3: Implement** the row builders in `gmst/features.py` and B1 in `gmst/baselines.py` (native `lightgbm.train`; import `backbone` inside `baselines`).
- [ ] **Step 4: Run to verify they pass** — `uv run pytest tests/test_features.py tests/test_baselines.py -q` → `32 passed` (19 in test_features, 13 in test_baselines).
- [ ] **Step 5: Full suite and commit**
```bash
git add gmst/features.py gmst/baselines.py tests/test_features.py tests/test_baselines.py
git commit -m "feat: B1 LightGBM (slot mean/quantiles, day M_d quantiles, peak classifiers) with inner-window state; A+W* oracle features"
```

---

### Task 9: Probability calibration (Platt, PAV), leave-one-fold-out, inner-window p*

**Files:**
- Modify: `gmst/evaluate.py` (append), `tests/test_evaluate.py` (append)

**Interfaces:**
- Produces (module `gmst.evaluate`):
  - `platt_fit(p, e, N=2000) -> tuple[float, float]` and `platt_apply(ab, p, N=2000) -> np.ndarray`
  - `pav_fit(x, y) -> tuple[np.ndarray, np.ndarray]` and `pav_apply(model, x) -> np.ndarray`
  - `fit_calibrator(rows, method) -> dict` (keys `"C50"`, `"C75"`, `"C90"`)
  - `apply_calibrator(cal, rows, method) -> pl.DataFrame`
  - `calibrate_oof(days, method="platt", inner=None, ref=None) -> tuple[pl.DataFrame, pl.DataFrame | None]`
  - `p_star(p, e) -> float` and `p_star_table(inner) -> dict[tuple[str, str, str], dict[str, float]]` keyed `(model, variant, fold)`

**Algorithm (v2 §11.3, FR-52…FR-55, A3, A24, ruling R10):**
- Platt: `x = logit(clip(p, 1/(2N), 1 − 1/(2N)))` (N = 2,000 → [0.00025, 0.99975]); maximise `Σ[e·log σ(a x + b) + (1−e)·log(1 − σ(a x + b))] − 0.5·1e-3·(a² + b²)` by Newton (start a = 1, b = 0, ≤ 100 iterations, stop when the max |step| < 1e-10). The tiny ridge keeps separable data finite. Empty input → `(1.0, 0.0)` (identity). `platt_apply` maps NaN → NaN.
- PAV: sort by x, merge equal x into weighted means, pool adjacent violators with weights; returns `xs` = sorted unique x and `ys` non-decreasing. `pav_apply = np.interp(x, xs, ys)` (flat outside), NaN → NaN. Empty input → `([0, 1], [0, 1])`. isotonic is an ablation (cut item ②).
- `fit_calibrator(rows, method)`: per threshold uses rows with `usable_peak` and finite `risk_raw_C*` and `event_C*`.
- `apply_calibrator`: writes `risk_{method}_C50/75/90`, then enforces `C50 ≥ C75 ≥ C90` row-wise with `np.minimum.accumulate` along the threshold axis (FR-50).
- `calibrate_oof`: for each `(model, variant)` group and each fold `j` of that group in `days`: if `ref is None`, the calibrator is fit on the group's rows in `days` with fold ≠ j (leave-one-fold-out, FR-54); if `ref` is given, it is fit on all of the group's rows in `ref` (used for the test fold with `ref = f1–f4 OOF`). Apply it to fold-j rows of `days` and to fold-j rows of `inner` (same model/variant). Returns `(days, inner)`.
- `p_star(p, e)`: candidates = sorted unique finite `p`; alarm = `p ≥ candidate`; F1 per candidate (`prf`); return the smallest candidate with maximal F1; return 0.5 if there are no events or no candidates.
- `p_star_table(inner)`: per `(model, variant, fold)`, per threshold, `p_star(risk_platt_C*, event_C*)` over `usable_peak` inner rows. p* for fold j therefore comes only from fold j's own inner window (never its validation days); the test fold's p* comes from 08-24…08-30 (**[plan pin]**, see Spec issues).

- [ ] **Step 1: Write the failing tests** — append to `tests/test_evaluate.py`:

```python
def test_platt_recovers_true_params():
    rng = np.random.default_rng(0)
    x = rng.uniform(-4, 4, 20000)
    e = rng.random(20000) < 1 / (1 + np.exp(-(1.5 * x - 0.5)))
    a, b = ev.platt_fit(1 / (1 + np.exp(-x)), e)
    assert a == pytest.approx(1.5, abs=0.1) and b == pytest.approx(-0.5, abs=0.1)


def test_platt_clipping_and_degenerate():
    ab = ev.platt_fit(np.array([0.0, 1.0, 0.0, 1.0]), np.array([0, 1, 0, 1]))
    out = ev.platt_apply(ab, np.array([0.0, 1.0, np.nan]))
    assert np.isfinite(out[:2]).all() and out[0] < out[1] and np.isnan(out[2])
    assert np.isfinite(ev.platt_apply(ev.platt_fit(np.array([0.2, 0.4]), np.array([0, 0])), np.array([0.3]))).all()
    assert ev.platt_fit(np.array([]), np.array([])) == (1.0, 0.0)


def test_pav():
    xs, ys = ev.pav_fit(np.array([1.0, 2, 3, 4]), np.array([1.0, 3, 2, 4]))
    assert ys.tolist() == pytest.approx([1, 2.5, 2.5, 4])
    assert ev.pav_apply((xs, ys), np.array([0.0, 2.5, 9.0])).tolist() == pytest.approx([1, 2.5, 4])
    rng = np.random.default_rng(0)
    x = rng.random(200)
    _, ys = ev.pav_fit(x, (rng.random(200) < x).astype(float))
    assert (np.diff(ys) >= -1e-12).all()


def cal_days(seed, flip_fold=None, folds=("f1", "f2", "f3", "f4")):
    rng = np.random.default_rng(seed)
    rows = []
    for f in folds:
        raw = rng.random(40)
        evt = rng.random(40) < raw
        if f == flip_fold:
            evt = ~evt
        for r, e in zip(raw, evt):
            rows.append({"fold": f, "variant": "main", "model": "M", "usable_peak": True,
                         **{f"risk_raw_C{c}": float(r) for c in (50, 75, 90)},
                         **{f"risk_{m}_C{c}": float("nan") for m in ("platt", "iso") for c in (50, 75, 90)},
                         **{f"event_C{c}": float(e) for c in (50, 75, 90)}})
    return pl.DataFrame(rows)


def test_calibrate_oof_leave_one_fold_out():
    a, _ = ev.calibrate_oof(cal_days(0), "platt")
    b, _ = ev.calibrate_oof(cal_days(0, flip_fold="f2"), "platt")
    col = lambda d, f: d.filter(pl.col("fold") == f)["risk_platt_C90"].to_numpy()
    assert np.array_equal(col(a, "f2"), col(b, "f2"))
    assert not np.array_equal(col(a, "f1"), col(b, "f1"))


def test_calibrated_risk_monotone_in_C():
    df = cal_days(3).with_columns(pl.lit(0.99).alias("risk_raw_C90"))
    for method in ("platt", "iso"):
        out, _ = ev.calibrate_oof(df, method)
        r = out.select(f"risk_{method}_C50", f"risk_{method}_C75", f"risk_{method}_C90").to_numpy()
        assert (r[:, 0] >= r[:, 1]).all() and (r[:, 1] >= r[:, 2]).all()


def test_inner_rows_use_own_fold_calibrator():
    inner = cal_days(1).filter(pl.col("fold") == "f3")
    _, a = ev.calibrate_oof(cal_days(0), "platt", inner)
    _, b = ev.calibrate_oof(cal_days(0, flip_fold="f3"), "platt", inner)
    assert np.array_equal(a["risk_platt_C90"].to_numpy(), b["risk_platt_C90"].to_numpy())
    assert np.isfinite(a["risk_platt_C90"].to_numpy()).all()


def test_reference_calibration_for_test_fold():
    ref = cal_days(0)
    test = cal_days(5, folds=("test",))
    out, _ = ev.calibrate_oof(test, "platt", ref=ref)
    ab = ev.platt_fit(ref["risk_raw_C50"].to_numpy(), ref["event_C50"].to_numpy())
    assert out["risk_platt_C50"].to_numpy() == pytest.approx(ev.platt_apply(ab, test["risk_raw_C50"].to_numpy()))


def test_p_star():
    assert ev.p_star(np.array([0.1, 0.2, 0.3, 0.8, 0.9]), np.array([0, 0, 1, 1, 1])) == pytest.approx(0.3)
    assert ev.p_star(np.array([0.4, 0.4]), np.array([0, 0])) == 0.5


def test_p_star_table():
    inner = pl.DataFrame({"fold": ["f1"] * 5, "variant": ["main"] * 5, "model": ["M"] * 5, "usable_peak": [True] * 5,
                          **{f"risk_platt_C{c}": [0.1, 0.2, 0.3, 0.8, 0.9] for c in (50, 75, 90)},
                          **{f"event_C{c}": [0.0, 0.0, 1.0, 1.0, 1.0] for c in (50, 75, 90)}})
    assert ev.p_star_table(inner)[("M", "main", "f1")] == pytest.approx({"C50": 0.3, "C75": 0.3, "C90": 0.3})
```

- [ ] **Step 2: Run to verify they fail** — `uv run pytest tests/test_evaluate.py -q` → `AttributeError: module 'gmst.evaluate' has no attribute 'platt_fit'`.
- [ ] **Step 3: Implement** the calibration functions (numpy Newton, numpy PAV).
- [ ] **Step 4: Run to verify they pass** — `uv run pytest tests/test_evaluate.py -q` → `24 passed`.
- [ ] **Step 5: Full suite and commit**
```bash
git add gmst/evaluate.py tests/test_evaluate.py
git commit -m "feat: Platt (clipped logit, Newton) and PAV calibration, leave-one-fold-out, inner-window p*"
```

---

### Task 10: Peak-risk discrimination check, climatology, day-block bootstrap, DM, B1-benchmark gate, B4-family selection

**Files:**
- Modify: `gmst/evaluate.py` (append), `tests/test_evaluate.py` (append)

**Interfaces:**
- Consumes: Tasks 5, 6, 9.
- Produces (module `gmst.evaluate`):
  - `climatology(M, op, dtype, C) -> dict` — `{"base": (3,), "cond": {(op, dtype): (3,)}, "op": {op: (3,)}}` event rates `M > C_j`.
  - `clim_prob(clim, op, dtype) -> np.ndarray` (3,) — `cond[(op, dtype)]` if present, else `op[op]`, else `base` (holidays not in the class, ruling Q8).
  - `add_climatology(days, panel) -> pl.DataFrame` — adds `clim_C50/75/90`, `climcond_C50/75/90`, `op_true` (Int8), `dtype` (Int8) per row; the fold's climatology uses that fold's `train` days with `usable_peak`, their daily max, true `op`, `dtype`, and the row's own `C50/C75/C90`.
  - `risk_metrics(days, panel, p_star) -> pl.DataFrame` — same long schema as `point_metrics`; metrics `brier_platt_C*`, `brier_iso_C*`, `brier_mean_platt`, `brier_mean_iso`, `brier_clim_C*`, `brier_climcond_C*`, `bss_cond_C*` (= 1 − Brier_platt / Brier_climcond), and for each `k ∈ {p05, pstar}`: `f1_C*_k`, `precision_C*_k`, `recall_C*_k`, `fn_C*_k`, `fp_C*_k` (alarm = `risk_platt ≥ 0.5` or `≥ p_star[(model, variant, fold)][C]`; pooled rows use each row's own fold p*). Folds, `pooled` and strata exactly as `point_metrics`. NaN F1/precision/recall when undefined (f4 q90: no events).
  - `risk_check(days) -> dict` — keyed `"<model>/<variant>"`; per `C50/C75/C90`: `brier` (Platt), `brier_climcond`, `auc`, `bss_cond`, `n_events`, `n_days`, `pass`; plus `"discrimination_missing": not all(pass)`. Uses rows with fold in `FOLDS_CV` and `usable_peak` (pooled f1–f4). `pass = brier < brier_climcond and auc >= 0.70 and bss_cond > 0`; when the pooled events are single-class, `pass = bss_cond > 0` and `auc` = NaN (FR-41, A26, v2 §11.3).
  - `bootstrap_days(panel) -> np.ndarray` — f1–f4 validation day indices that are not `is_suspect` (ruling R5: 54 days; the fully masked copy day 07-30 stays in the set with zero weight — see Spec issues).
  - `day_losses(slots, days, panel) -> pl.DataFrame` — one row per bootstrap day: `fold, date, ae_sum, n_pts, crps_sum, n_crps, brier_sum, n_brier` (`brier_sum` = mean over the three thresholds of `(risk_platt − event)²` on a `usable_peak` day else 0; `n_brier` = 1 or 0).
  - `block_bootstrap(num, den, B=2000, seed=0) -> tuple[float, float, float]` — statistic `Σnum/Σden`; `rng = np.random.default_rng(seed)`; `idx = rng.integers(0, n, size=(B, n))`; replicate ratios; `(stat, nanpercentile 2.5, nanpercentile 97.5)`.
  - `dm_test(d) -> tuple[float, float]` — finite daily differentials; `stat = mean / (sd_ddof1 / sqrt(n))`; `p = math.erfc(abs(stat) / sqrt(2))`; `sd == 0` → `stat = 0.0` if mean is 0 else `±inf`.
  - `compare(slots_a, days_a, slots_b, days_b, panel, name, metrics=("mae", "crps", "brier_mean"), B=2000, seed=0) -> list[dict]` — rows `{comparison, metric, delta, ci_lo, ci_hi, dm_stat, dm_p, n_days}` for `a − b` (FR-56). `mae`: num = `ae_sum_a − ae_sum_b`, den = `n_pts` (assert equal masks); `crps`: `crps_sum` diff over `n_crps`; `brier_mean`: `brier_sum` diff over `n_brier`. DM series = per-day `num/den` where `den > 0`. The resampled day set = `bootstrap_days(panel)` restricted to days present in both frames; `n_days` = its size (54 in the full run, 14 in the f4-only smoke run). Marker in this function: `# ponytail: 일 블록 부트스트랩, 요일 간 상관이 크면 7일 블록 (A13)`.
  - `gate_pass(boot, comparison) -> bool` — the rows of `boot` with that `comparison` (e.g. `"B4-B1"`, `"B4-H-B1"`); `True` iff the `mae` and `brier_mean` rows both have `ci_hi < 0` (US-008 as amended, FR-57). This is the **success criterion** of the B4 family against the B1 benchmark, not a switch: B1 is never submitted (user decision). CRPS never enters (ruling Q5). (**[plan pin]**: the PRD writes `gate_pass(variant_metrics, b1_metrics)`; the plan passes the bootstrap rows that already hold both.)
  - `select_variant(candidates: dict[str, dict]) -> dict` (US-021, FR-110) — `candidates` maps variant IDs in tried order (`"main"`, `"H"`, `"IO"`, `"KAN"`) to `{"mae": float, "brier_mean": float, "gate_met": bool, "gap": {"mae": [d, lo, hi], "brier_mean": [d, lo, hi]}}`. Pool = the candidates with `gate_met` if any, else all; winner = lowest `mae`, ties by lowest `brier_mean`, then tried order. Returns `{"submitted": <variant ID>, "gate_met": bool (the winner's), "tried": [IDs in order], "remaining_gap_to_b1": <winner's gap>}`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_evaluate.py`:

```python
from gmst import baselines as bl


def test_climatology_conditional():
    M = np.array([200.0, 150.0, 210.0, 100.0, 190.0, 300.0])
    op = np.array([1, 1, 1, 0, 1, 1])
    dt = np.array([0, 0, 1, 0, 1, 2])
    cl = ev.climatology(M, op, dt, np.array([180.0, 195.0, 205.0]))
    assert cl["base"] == pytest.approx([4 / 6, 3 / 6, 2 / 6])
    assert ev.clim_prob(cl, 1, 0) == pytest.approx([0.5, 0.5, 0.0])
    assert ev.clim_prob(cl, 1, 1) == pytest.approx([1.0, 0.5, 0.5])
    assert ev.clim_prob(cl, 0, 2) == pytest.approx([0.0, 0.0, 0.0])      # empty class -> op-only rate
    assert ev.clim_prob(cl, 1, 2) == pytest.approx([1.0, 1.0, 1.0])


def risk_days(p, e):
    n = len(e)
    return pl.DataFrame({"fold": ["f1"] * (n // 2) + ["f2"] * (n - n // 2), "variant": ["main"] * n,
                         "model": ["M"] * n, "usable_peak": [True] * n,
                         **{f"risk_platt_C{c}": p for c in (50, 75, 90)},
                         **{f"event_C{c}": e for c in (50, 75, 90)},
                         **{f"climcond_C{c}": [0.5] * n for c in (50, 75, 90)}})


def test_risk_check_pass_and_fail():
    e = [0.0, 1.0] * 10
    good = ev.risk_check(risk_days([0.1, 0.9] * 10, e))["M/main"]
    assert good["C90"]["pass"] and not good["discrimination_missing"]
    flat = ev.risk_check(risk_days([0.5] * 20, e))["M/main"]
    assert flat["C90"]["auc"] == pytest.approx(0.5) and flat["discrimination_missing"]
    one = ev.risk_check(risk_days([0.1] * 20, [0.0] * 20))["M/main"]["C90"]
    assert np.isnan(one["auc"]) and one["pass"] == (one["bss_cond"] > 0) and one["n_events"] == 0


def test_block_bootstrap_ci_contains_truth():
    rng = np.random.default_rng(0)
    den = rng.integers(50, 97, 60).astype(float)
    eps = rng.normal(0, 1, 60)
    eps -= (den * eps).sum() / den.sum()                  # weighted mean exactly 0 -> statistic is exactly -2
    num = den * (-2.0 + eps)
    d, lo, hi = ev.block_bootstrap(num, den, B=2000, seed=0)
    assert d == pytest.approx(-2.0) and lo < -2.0 < hi
    assert ev.block_bootstrap(num, den, B=2000, seed=0) == (d, lo, hi)


def test_dm_test():
    stat, p = ev.dm_test(np.array([1.0, -1.0] * 50))
    assert stat == pytest.approx(0.0) and p == pytest.approx(1.0)
    stat, p = ev.dm_test(np.array([6.0, 4.0] * 50))
    assert stat > 10 and p < 1e-10


@pytest.mark.parametrize("mae_hi,brier_hi,met", [(-0.1, -0.01, True), (0.1, -0.01, False),
                                                 (-0.1, 0.01, False), (0.1, 0.01, False)])
def test_gate_pass(mae_hi, brier_hi, met):
    boot = pl.DataFrame({"comparison": ["B4-B1"] * 3 + ["AWstar-main"],
                         "metric": ["mae", "brier_mean", "crps", "mae"],
                         "delta": [-1.0, -0.02, -5.0, 9.0], "ci_lo": [-2.0, -0.05, -9.0, 8.0],
                         "ci_hi": [mae_hi, brier_hi, -1.0, -10.0],
                         "dm_stat": [0.0] * 4, "dm_p": [1.0] * 4, "n_days": [54] * 4})
    assert ev.gate_pass(boot, "B4-B1") is met


def cand(mae, brier, met):
    return {"mae": mae, "brier_mean": brier, "gate_met": met, "gap": {"mae": [mae - 12.0, -1.0, 1.0],
                                                                     "brier_mean": [0.0, -0.01, 0.01]}}


def test_select_variant():
    out = ev.select_variant({"main": cand(12.0, 0.10, False), "H": cand(11.0, 0.12, True), "IO": cand(10.0, 0.11, False)})
    assert out["submitted"] == "H" and out["gate_met"] is True and out["tried"] == ["main", "H", "IO"]
    assert out["remaining_gap_to_b1"]["mae"][0] == pytest.approx(-1.0)
    out = ev.select_variant({"main": cand(12.0, 0.10, False), "H": cand(11.0, 0.12, False), "IO": cand(11.0, 0.11, False)})
    assert out["submitted"] == "IO" and out["gate_met"] is False
    out = ev.select_variant({"main": cand(12.0, 0.10, True), "H": cand(11.5, 0.12, True)})
    assert out["submitted"] == "H"
    assert ev.select_variant({"main": cand(12.0, 0.10, False)})["submitted"] == "main"


def test_bootstrap_day_set_matches_day_table(panel):
    idx = ev.bootstrap_days(panel)
    days = panel["days"]
    expected = [int(i) for f in ev.FOLDS_CV for i in ft.role_idx(panel, f, "val") if not days["is_suspect"][int(i)]]
    assert sorted(idx.tolist()) == sorted(expected)
    assert len(expected) == 54            # ruling R5 / FR-56 / v2 A13


def test_compare_rows(panel):
    a = ev.rolling_origin(bl.b0_model(), panel)
    b = ev.rolling_origin(bl.b0p_model(), panel)
    rows = ev.compare(a[0], a[1], b[0], b[1], panel, "B0-B0p", metrics=("mae",), B=200)
    assert rows[0]["comparison"] == "B0-B0p" and rows[0]["metric"] == "mae" and rows[0]["n_days"] == 54
    mae_a = ev.mae(a[0]["y_true"].to_numpy(), a[0]["y_median"].to_numpy())[0]
    mae_b = ev.mae(b[0]["y_true"].to_numpy(), b[0]["y_median"].to_numpy())[0]
    assert rows[0]["delta"] == pytest.approx(mae_a - mae_b)
    assert rows[0]["ci_lo"] <= rows[0]["delta"] <= rows[0]["ci_hi"]


def test_risk_metrics_names(panel):
    _, days, _, _ = ev.rolling_origin(bl.b0p_model(), panel)
    days = days.with_columns([pl.col(f"risk_raw_C{c}").alias(f"risk_{m}_C{c}")
                              for m in ("platt", "iso") for c in (50, 75, 90)])
    ps = {("B0p", "main", f): {"C50": 0.5, "C75": 0.5, "C90": 0.5} for f in ev.FOLDS_CV}
    m = ev.risk_metrics(ev.add_climatology(days, panel), panel, ps)
    names = set(m["metric"].unique())
    for c in ("C50", "C75", "C90"):
        assert {f"brier_platt_{c}", f"brier_iso_{c}", f"brier_clim_{c}", f"brier_climcond_{c}", f"bss_cond_{c}",
                f"f1_{c}_p05", f"f1_{c}_pstar", f"precision_{c}_pstar", f"recall_{c}_p05", f"fn_{c}_pstar",
                f"fp_{c}_p05"} <= names
    assert {"brier_mean_platt", "brier_mean_iso"} <= names
    f4 = m.filter((pl.col("fold") == "f4") & (pl.col("stratum") == "all") & (pl.col("metric") == "f1_C90_pstar"))
    assert np.isnan(f4["value"][0])
```

- [ ] **Step 2: Run to verify they fail** — `uv run pytest tests/test_evaluate.py -q` → `AttributeError: module 'gmst.evaluate' has no attribute 'climatology'`.
- [ ] **Step 3: Implement** the Task 10 functions.
- [ ] **Step 4: Run to verify they pass** — `uv run pytest tests/test_evaluate.py -q` → `36 passed`.
- [ ] **Step 5: Full suite and commit**
```bash
git add gmst/evaluate.py tests/test_evaluate.py
git commit -m "feat: conditional climatology, risk discrimination check (AUC/BSS), day-block bootstrap, DM, B1 gate, B4-family selection"
```

---

### Task 11: Conditional HMM core (pairwise-emission forward, smoothing, block training, GPU/CPU)

**Files:**
- Create: `gmst/hmm.py`, `tests/test_hmm.py`

**Interfaces:**
- Consumes: `features.cal_flags`, `features.role_idx`, `features.inner_idx`.
- Produces (module `gmst.hmm`):
  - `DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")`; at import nothing is printed (run_all logs the device).
  - `DZ = {"A": 8, "A+": 14, "B": 16}`
  - `z_features(panel, protocol, day_idx) -> np.ndarray` (n, 96, dz) float64. Column order — A+: `[sin1, cos1, sin2, cos2, sin3, cos3, op, op·sin1, op·cos1, op·sin2, op·cos2, sat, sun, hol]`; A: `[sin1, cos1, sin2, cos2, sin3, cos3, sat, sun]`; B: A+ + `[1(P > 0), log1p(P)/10]` with `P = X["생산량"][d, q]` (NaN → 0 with marker `# ponytail: 결측 생산량을 전이 특징에서 0으로, suspect일 전이가 결과를 좌우하면 결측 지시변수 추가 (FR-63)`). `op, hol` = `cal_flags(panel, protocol)`; `sat = dtype == 1`, `sun = dtype == 2`; `sin_r/cos_r = sin/cos(2π r q / 96)` (FR-63, A15, A20).
  - `class CondHMM(torch.nn.Module)` — `__init__(self, K=3, dz=14, cond=True, ar=True, seed=0)`. Parameters (exact names, trainable):
    - `W`: (K, K−1, 1+dz) if `cond` else (K, K−1, 1). Row i, column c holds the logits of the off-diagonal targets `j ≠ i` in increasing j; index 0 of the last axis is the intercept `b_ij`. Init: intercept −3.0, all entries + N(0, 0.1²).
    - `rho`: (K, 2) — `δ_{1,op} = rho[0, op]`, `δ_{k,op} = δ_{k−1,op} + softplus(rho[k−1, op])`. Init from δ = linspace(−20, 20, K) (op = 1) and linspace(−2, 2, K) (op = 0) via inverse softplus, + N(0, 0.5²).
    - `s`: (K, 2) — `σ_{k,op} = 1 + softplus(s[k, op])` (floor ε = 1, DC16). Init σ = 10 (op 1), 2 (op 0) + N(0, 0.1²) on `s`.
    - `psi`: (K,) only when `ar` — `φ_k = sigmoid(psi_k)`; init 1.0 + N(0, 0.1²). When `ar=False`, `φ ≡ 0` and no `psi`.
    - Init noise uses `torch.Generator().manual_seed(seed)` so seeds differ only through init.
    - Methods: `trans(z) -> A` (…, K, K) — `A_t(i, ·) = softmax` over `j` of the logits with the `j = i` logit fixed at 0 (FR-64); `z` shape (…, dz) (ignored when `cond=False`). `delta() -> (K, 2)`, `sigma() -> (K, 2)`, `phi() -> (K,)`. `emission(m, opt) -> (mu, sig)` with `mu[..., k] = m + δ[k, opt]` and `sig[..., k] = σ[k, opt]` (FR-65).
    - Trainable parameter counts: K(K−1)(1+dz) + 5K (4K when `ar=False`; K(K−1) transitions when `cond=False`) (FR-64, FR-65).
  - `stationary(A) -> π` (…, K) with `π A = π`, `Σπ = 1` (linear solve of `(Aᵀ − I)` with its last row replaced by ones), differentiable.
  - `forward_logp(model, y, obs, m, opt, z) -> tuple[Tensor, Tensor]` — inputs (B, T) / (B, T, dz); returns `logc` (B, T) and filtered `alpha` (B, T, K). Algorithm (FR-66, FR-67, v2 §10): replace unobserved `y` by 0 before use; log-emission `lb[t, i, k]` = `log N(y_t; μ_{t,k} + φ_k (y_{t−1} − μ_{t−1,i}), σ²_{t,k})` when `obs[t]` and `obs[t−1]` and `t > 0`; `log N(y_t; μ_{t,k}, σ²_{t,k}/(1 − φ_k²))` when `obs[t]` and (`t = 0` or not `obs[t−1]`); 0 when not `obs[t]`. Prior `α̃_0 = stationary(A_0) ⊙ exp(lb[0, 0, :])`. For t ≥ 1: `s_t = max_{i,k} lb[t]`, `α̃_t(k) = Σ_i α_{t−1}(i) A_t(i,k) exp(lb[t,i,k] − s_t)`, `c_t = Σ_k α̃_t(k)`, `logc_t = log c_t + s_t`, `α_t = α̃_t / c_t` (same shift at t = 0). Precompute `A` and `lb` for all t in one shot; only the recursion loops over t.
  - `backward(model, y, obs, m, opt, z) -> gamma` (B, T, K) — smoothed marginals `P(S_t = k | y_{1:T})` via the scaled backward pass `β_{t−1}(i) = Σ_k A_t(i,k) exp(lb[t,i,k] − logc_t) β_t(k)`, `β_{T−1} = 1`, `γ = α β` renormalised per step.
  - `make_blocks(panel, day_idx, m, protocol, device="cpu", dtype=torch.float32) -> dict | None` — one block per `d ∈ day_idx` with `d ≥ 1` and ≥ 1 finite `Y[d]`: keys `y, obs, m, opt, z` over 192 steps (day d−1 then day d; `opt` = `op_eff` of each step's day), `days` (np.ndarray of target indices). `None` when empty (FR-68).
  - `nll(model, blocks) -> Tensor` = `−logc[:, 96:].sum() / obs[:, 96:].sum()` (per observed target slot, v2 §14).
  - `occupancy(model, blocks) -> Tensor` (K,) = mean filtered `alpha` over target steps; `occ_penalty(occ, kappa=0.02, lam=10.0) -> Tensor` = `lam · Σ relu(kappa − occ)²` (FR-68, A21).
  - `fit_blocks(model, train, val=None, max_epochs=300, patience=20, epochs=None, lr=1e-2, occ_floor=False) -> tuple[model, int, list[dict]]` — full-batch Adam; each epoch: loss = `nll(train)` (+ `occ_penalty(occupancy(train))` if `occ_floor`), step; history row `{"epoch", "train_nll", "val_nll"}`. Early-stopping mode (`epochs is None`, `val` required): after each step compute `nll(val)` under `no_grad`; keep a deep copy of the best state (strictly lower val NLL); stop when `epoch − best_epoch ≥ patience` or at `max_epochs`; restore the best state; return `best_epoch` (≥ 1). Fixed mode: exactly `epochs` steps, `best_epoch = epochs`, `val_nll` NaN.
  - `train_hmm(panel, train_idx, val_idx, m, protocol="A+", K=3, cond=True, ar=True, seed=0, max_epochs=300, patience=20, epochs=None, occ_floor=False, device=DEVICE) -> tuple[model, int, list[dict]]` — `torch.manual_seed(seed)`, builds `CondHMM(K, DZ[protocol], cond, ar, seed)` on `device`, blocks from `train_idx` / `val_idx`, calls `fit_blocks`.
  - `fit_with_inner(panel, train_idx, inner, m_in, m_full, epochs=None, **kw) -> tuple[inner_model, model, int, list[dict]]` (ruling R10) — `epochs is None`: `inner_model, best, hist = train_hmm(panel, setdiff(train_idx, inner), inner, m_in, **kw)` (early stopping on the inner window), then `model = train_hmm(panel, train_idx, [], m_full, epochs=best, **kw)[0]` (refit on the full train window); fixed `epochs`: both fits use `epochs`, `best = epochs`.
  - `filter_last(model, panel, d, m, protocol="A+") -> tuple[Tensor, float]` — forward over day d−1's 96 slots (prior = stationary of its first transition matrix); returns the last filtered distribution and `y_last = Y[d−1, 95]` (NaN if masked). If day d−1 has no observed slot → the stationary prior (FR-69, A9).
  - `posthoc_states(state, panel, d) -> tuple[np.ndarray, np.ndarray]` — 1-based argmax of filtered and smoothed marginals over day d using the block (d−1, d) with the actual `Y[d]` (FR-70; analysis only, never used by `predict`).

- [ ] **Step 1: Write the failing test** — create `tests/test_hmm.py`:

```python
import copy
import itertools
import math
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest
import torch

from gmst import features as ft
from gmst import hmm

ROOT = Path(__file__).resolve().parents[1]
Y3 = np.array([1.3, 0.2, 2.1])
M3 = np.array([0.5, -0.2, 1.0])
OP3 = np.array([1, 1, 0])
Z3 = np.array([[0.1, -0.3], [0.7, 0.2], [-0.5, 0.9]])


def rand_model(K=2, dz=2, cond=True, ar=True, seed=0, scale=1.0):
    model = hmm.CondHMM(K=K, dz=dz, cond=cond, ar=ar, seed=seed).double()
    g = torch.Generator().manual_seed(seed + 100)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(scale * torch.randn(p.shape, generator=g, dtype=p.dtype))
    return model


def tens(y, obs):
    return [torch.as_tensor(a)[None] for a in (y, obs, M3, OP3, Z3)]


def enumerate_paths(model, y, obs):
    A = model.trans(torch.as_tensor(Z3)).detach().numpy()
    mu, sig = (t.detach().numpy() for t in model.emission(torch.as_tensor(M3), torch.as_tensor(OP3)))
    phi = model.phi().detach().numpy()
    K = A.shape[1]
    pi = hmm.stationary(torch.as_tensor(A[0])).detach().numpy()
    npdf = lambda x, m, s: math.exp(-0.5 * ((x - m) / s) ** 2) / (s * math.sqrt(2 * math.pi))

    def b(t, i, k):
        if not obs[t]:
            return 1.0
        if t == 0 or not obs[t - 1]:
            return npdf(y[t], mu[t, k], sig[t, k] / math.sqrt(1 - phi[k] ** 2))
        return npdf(y[t], mu[t, k] + phi[k] * (y[t - 1] - mu[t - 1, i]), sig[t, k])

    out = []
    for s in itertools.product(range(K), repeat=len(y)):
        p = pi[s[0]] * b(0, None, s[0])
        for t in range(1, len(y)):
            p *= A[t, s[t - 1], s[t]] * b(t, s[t - 1], s[t])
        out.append((s, p))
    return out


@pytest.mark.parametrize("K,dz,cond,ar,n", [(3, 14, True, True, 105), (3, 8, True, True, 69), (3, 16, True, True, 117),
                                            (2, 14, True, True, 40), (4, 14, True, True, 200),
                                            (3, 14, False, False, 18), (3, 14, True, False, 102)])
def test_param_counts(K, dz, cond, ar, n):
    model = hmm.CondHMM(K=K, dz=dz, cond=cond, ar=ar)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == n


def test_z_features():
    p = ft.load_panel()
    idx = np.array([p["dates"].index(date(2021, 7, 10)), p["dates"].index(date(2021, 8, 16))])
    zA, zAp, zB = (hmm.z_features(p, pr, idx) for pr in ("A", "A+", "B"))
    assert zA.shape == (2, 96, 8) and zAp.shape == (2, 96, 14) and zB.shape == (2, 96, 16)
    assert zAp[0, 24, 0] == pytest.approx(1.0) and zAp[0, 0, 1] == pytest.approx(1.0)
    assert (zAp[0, :, 6] == p["op"][idx[0]]).all()
    assert (zAp[0, 0, 11], zAp[0, 0, 12], zAp[1, 0, 13]) == (1.0, 0.0, 1.0)     # 07-10 Saturday, 08-16 holiday
    P = p["X"]["생산량"][idx[1]]
    assert np.allclose(zB[1, :, 14], P > 0) and np.allclose(zB[1, :, 15], np.log1p(P) / 10)


@pytest.mark.parametrize("obs", [(True, True, True), (True, False, True), (False, True, True)])
def test_forward_matches_bruteforce(obs):
    model = rand_model()
    obs = np.array(obs)
    logc, alpha = hmm.forward_logp(model, *tens(Y3, obs))
    assert logc.shape == (1, 3) and alpha.shape == (1, 3, 2)
    brute = math.log(sum(p for _, p in enumerate_paths(model, Y3, obs)))
    assert logc.sum().item() == pytest.approx(brute, abs=1e-6)


def test_masked_value_does_not_matter():
    model = rand_model()
    obs = np.array([True, False, True])
    y_other = Y3.copy()
    y_other[1] = 999.0
    assert torch.equal(hmm.forward_logp(model, *tens(Y3, obs))[0], hmm.forward_logp(model, *tens(y_other, obs))[0])


def test_backward_smoothing_bruteforce():
    model = rand_model()
    obs = np.array([True, True, True])
    gamma = hmm.backward(model, *tens(Y3, obs))[0].detach().numpy()
    paths = enumerate_paths(model, Y3, obs)
    total = sum(p for _, p in paths)
    for t in range(3):
        for k in range(2):
            assert gamma[t, k] == pytest.approx(sum(p for s, p in paths if s[t] == k) / total, abs=1e-6)


def test_state_order_and_sigma_floor():
    model = rand_model(K=3, dz=14, scale=3.0)
    mu, sig = model.emission(torch.zeros(10, dtype=torch.float64), torch.tensor([0, 1] * 5))
    assert (mu[:, 1:] > mu[:, :-1]).all() and (sig >= 1).all()
    d = model.delta()
    assert d.shape == (3, 2) and (d[1:] > d[:-1]).all()


def test_stationary():
    A = torch.softmax(torch.randn(4, 3, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(0)), -1)
    pi = hmm.stationary(A)
    assert pi.shape == (4, 3)
    assert torch.allclose(torch.einsum("bi,bij->bj", pi, A), pi, atol=1e-6)
    assert torch.allclose(pi.sum(-1), torch.ones(4, dtype=torch.float64))


def test_blocks_f1():
    p = ft.load_panel()
    blk = hmm.make_blocks(p, ft.role_idx(p, "f1", "train"), np.zeros((257, 96)), "A+")
    assert blk["y"].shape == (65, 192) and blk["z"].shape == (65, 192, 14)
    assert int(blk["obs"][:, 96:].sum()) == 6240


def synthetic_blocks(n=30, seed=0, dz=14):
    g = torch.Generator().manual_seed(seed)
    level = torch.tensor(([10.0] * 48 + [-10.0] * 48) * 2)
    y = level + torch.randn(n, 192, generator=g)
    return {"y": y, "obs": torch.ones(n, 192, dtype=torch.bool), "m": torch.zeros(n, 192),
            "opt": torch.ones(n, 192, dtype=torch.long), "z": torch.zeros(n, 192, dz), "days": np.arange(n)}


def test_nll_is_mean_over_observed_target_slots():
    blk = synthetic_blocks()
    blk["obs"][:, 150:160] = False
    model = hmm.CondHMM(K=2, dz=14)
    logc, _ = hmm.forward_logp(model, blk["y"], blk["obs"], blk["m"], blk["opt"], blk["z"])
    manual = -(logc[:, 96:].sum() / blk["obs"][:, 96:].sum())
    assert hmm.nll(model, blk).item() == pytest.approx(manual.item(), rel=1e-6)


def test_training_reduces_nll():
    blk = synthetic_blocks()
    model = hmm.CondHMM(K=2, dz=14, seed=0)
    before = hmm.nll(model, blk).item()
    model, best, hist = hmm.fit_blocks(model, blk, None, epochs=30)
    assert hmm.nll(model, blk).item() < before and best == 30 and len(hist) == 30


def test_early_stopping_restores_best():
    train, val = synthetic_blocks(seed=0), synthetic_blocks(seed=1)
    model, best, hist = hmm.fit_blocks(hmm.CondHMM(K=2, dz=14, seed=0), train, val, max_epochs=15, patience=3)
    vals = [h["val_nll"] for h in hist]
    assert best == int(np.argmin(vals)) + 1 and len(hist) <= 15
    assert hmm.nll(model, val).item() == pytest.approx(min(vals), rel=1e-6)


def test_device_cpu_when_hidden():
    out = subprocess.run([sys.executable, "-c", "from gmst.hmm import DEVICE; print(DEVICE)"], cwd=ROOT,
                         env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "cpu"
    assert hmm.DEVICE.type == ("cuda" if torch.cuda.is_available() else "cpu")


def test_fixed_epoch_training_ignores_later_values():
    p = ft.load_panel()
    tr = ft.role_idx(p, "f1", "train")
    m = np.zeros((257, 96))
    a = hmm.train_hmm(p, tr, np.array([], int), m, epochs=3, device="cpu")[0]
    p2 = {**p, "Y": p["Y"].copy()}
    p2["Y"][int(tr.max()) + 1:] = 123.0
    b = hmm.train_hmm(p2, tr, np.array([], int), m, epochs=3, device="cpu")[0]
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))


def test_occ_floor_penalty():
    assert hmm.occ_penalty(torch.tensor([0.0, 0.5, 0.5])).item() == pytest.approx(10 * 0.02 ** 2)
    assert hmm.occ_penalty(torch.tensor([0.3, 0.3, 0.4])).item() == 0.0
```

- [ ] **Step 2: Run to verify it fails** — `uv run pytest tests/test_hmm.py -q` → `ModuleNotFoundError: No module named 'gmst.hmm'`.
- [ ] **Step 3: Implement** the Task 11 part of `gmst/hmm.py`. Keep everything dtype-generic (tests run the model in float64 on CPU; production runs float32 on `DEVICE`).
- [ ] **Step 4: Run to verify it passes** — `uv run pytest tests/test_hmm.py -q` → `22 passed`; also `CUDA_VISIBLE_DEVICES="" uv run pytest tests/test_hmm.py -q` → `22 passed`.
- [ ] **Step 5: Full suite and commit**
```bash
git add gmst/hmm.py tests/test_hmm.py
git commit -m "feat: conditional HMM with ordered deviation states, pairwise AR(1) emission, scaled forward/backward, block training"
```

---

### Task 12: MC peak risk, analytic φ=0 risk, model ladder B2/B3/B4, K ablations, day random-effect diagnostic

**Files:**
- Modify: `gmst/hmm.py` (append), `tests/test_hmm.py` (append)

**Interfaces:**
- Consumes: Task 11, `backbone.fold_backbone`, `features.role_idx`, `features.inner_idx`, `evaluate.rolling_origin`, `evaluate.thresholds`, `evaluate.TAUS`.
- Produces (module `gmst.hmm`):
  - `forecast(model, panel, d, m, C, protocol="A+", N=2000, re=False, s_op=None, seed=None, device=None) -> dict` — prediction dict with `paths` (N, 96) float64 numpy. `device` defaults to the model's device; `seed` defaults to `int(panel["dates"][d].strftime("%Y%m%d"))`.
  - `analytic_risk(model, panel, d, m, C, protocol="A+", device=None) -> np.ndarray` (3,)
  - `sigma_re(sigma, phi, s) -> np.ndarray`
  - `day_resid_sd(panel, train_idx, m) -> np.ndarray` (2,) indexed by op
  - `final_epochs(best: list[int]) -> int` = `int(np.floor(np.median(best) + 0.5))`
  - `hmm_model(kind="B4", protocol="A+", K=3, re=False, tau=10.0, half_life=60, final_epochs=None, max_epochs=300, patience=20, N=2000, train_from=None, occ_floor=False, seed=0, device=DEVICE) -> dict` — name = `kind`; keys `name, fit, predict, posthoc`. (Tasks 21–23 add the improvement-loop arguments `emission_source`, `decoder`, `kan`.)

**Algorithm (v2 §8–§11, FR-69…FR-75, A8–A10, A30):**
- `forecast`: `pi, y_last = filter_last(model, panel, d, m, protocol)`; transition matrices `A_h` from `z_features(panel, protocol, [d])` (h = 0…95 is the transition into slot h of day d); `mu, sig` for day d (op_eff[d]) and `mu_last = m[d−1, 95] + δ[:, op_eff[d−1]]`. Draw, in this fixed order from one `torch.Generator(device).manual_seed(seed)`: `u0` (N) uniform, `e0n` (N) normal, `u` (N, 96) uniform, `eps` (N, 96) normal, `ure` (N) normal — always all five, so the same seed gives common random numbers across scenarios. `S_0` = inverse-CDF of `pi` at `u0`; `e_0 = y_last − mu_last[S_0]` if `y_last` is finite else `e0n · σ_{S_0,op(d−1)} / sqrt(1 − φ²_{S_0})` (FR-71, 수정-5); for h: `S_h` = inverse-CDF of `A_h[S_{h−1}]` at `u[:, h]`, `e_h = φ_{S_h} e_{h−1} + σ_{S_h,op(d)} eps[:, h]`, `Y_h = μ_{h,S_h} + e_h`. With `re=True` (the §11.5 **diagnostic** ablation, FR-75, A30 — not a loop step; B4-IO subsumes its role): every σ is replaced by `sigma_re(σ, φ, s_op[op(d)])` and `u_d = ure · s_op[op(d)]` is added to all 96 slots of each path. Summaries on CPU numpy float64: `y_mean = paths.mean(0)`, `y_median = np.median(paths, 0)`, `q = np.quantile(paths, TAUS, axis=0)`, `M = paths.max(1)`, `M_hat_median = np.median(M)`, `M_hat_mean = M.mean()`, `peak_time_mode = np.bincount(paths.argmax(1), minlength=96).argmax()` (earliest on ties), `risk_raw = (M[:, None] > C).mean(0)`; if `model.phi()` is identically 0 (B2/B3) `risk_raw = analytic_risk(...)` instead (FR-72).
- `analytic_risk` (v2 §11.2): `P(M ≤ C) = π · Π_h [A_h · diag(Φ((C − μ_{h,k}) / σ_{k,op(d)}))] · 1`, computed in float64 on CPU with per-step renormalisation (accumulate the log of each step's sum; a zero sum gives probability 0); returns `1 − P` for the three thresholds. Only valid for φ = 0.
- `sigma_re(sigma, phi, s)`: `v = σ² / (1 − φ²)`, `v' = max(v − s², 1)`, `σ' = sqrt(v' (1 − φ²))` (elementwise). Marker: `# ponytail: 적률 분해로 σ 재추정, re 결과가 보고서 결론을 좌우하면 u_d를 넣은 우도로 재학습 (FR-75)`.
- `day_resid_sd(panel, train_idx, m)`: for train days with ≥ 1 finite Y: `r_d = nanmean(Y[d] − m[d])`; per true op `o`, `std(r_d, ddof=1)` (0.0 when fewer than 2 days).
- `hmm_model(kind, …)`: `kind` → `(cond, ar)` = B2 `(False, False)`, B3 `(True, False)`, B4 `(True, True)`. `fit(panel, fold, C)`: `tr = role_idx(panel, fold, "train")`; if `train_from` is set, `assert fold in ("f2", "f3", "f4")` and keep `tr` dates ≥ `train_from` (SCENARIO, FR-87); `inn = intersect(inner_idx(panel, fold), tr)`; `m_full = fold_backbone(panel, tr, tau, half_life, protocol)`, `m_in = fold_backbone(panel, setdiff(tr, inn), …)`; if `fold == "test"`: `assert final_epochs is not None` and `fit_with_inner(..., epochs=final_epochs)`, else early stopping (`max_epochs`, `patience`). State keys: `model, m, C, s_op (day_resid_sd on tr with m_full), best_epoch, history, val_nll (nll of the refit model on the fold's val blocks; NaN for test), device, protocol, kind, K, train_idx (= tr), inner_state` where `inner_state = {"model": inner_model, "m": m_in, "C": C, "s_op": day_resid_sd(setdiff(tr, inn), m_in), "device": device}`. `predict(state, panel, d)` = `forecast(state["model"], panel, d, state["m"], state["C"], protocol, N, re, state["s_op"], device=state["device"])`. `posthoc = posthoc_states`.
- Variants run later (Task 18): `K2`, `K4` (`hmm_model("B4", K=2|4)`), `re` (`hmm_model("B4", re=True)` reusing the B4 main states through `rolling_origin(states=…)`, no retraining; diagnostic only), `copies`/`suspect` (other panels), `protoA` (`protocol="A"`), `B` (SCENARIO: `protocol="B"`, `train_from=date(2021, 7, 1)`, folds f2–f4). §11.5 order (v2 as amended): check B3, B4 (and B4-H when it exists) with `risk_check` → if discrimination is missing, read the `re` diagnostic → continue the improvement loop (B4-IO, B4-KAN, Tasks 22–23). B1 is never a fallback.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_hmm.py`:

```python
from gmst import evaluate as ev


@pytest.fixture(scope="module")
def fitted():
    p = ft.load_panel()
    model = hmm.hmm_model("B4", tau=10.0, half_life=60, max_epochs=3, N=200, device="cpu")
    C, _ = ev.thresholds(p, ft.role_idx(p, "f1", "train"))
    return p, model, model["fit"](p, "f1", C)


def test_forecast_contract(fitted):
    p, model, state = fitted
    d = p["dates"].index(date(2021, 7, 7))
    a, b = model["predict"](state, p, d), model["predict"](state, p, d)
    assert a["paths"].shape == (200, 96) and a["q"].shape == (19, 96)
    assert (np.diff(a["q"], axis=0) >= 0).all() and 0 <= a["peak_time_mode"] <= 95
    assert a["risk_raw"][0] >= a["risk_raw"][1] >= a["risk_raw"][2]
    for k in ("y_mean", "y_median", "q", "paths", "risk_raw"):
        assert np.array_equal(a[k], b[k])


def toy_panel():
    return {"dates": [date(2021, 7, 5), date(2021, 7, 6)], "Y": np.zeros((2, 96)),
            "X": {k: np.zeros((2, 96)) for k in ("생산량", "기온", "풍속", "습도", "강수량_증분")},
            "op": np.array([1, 1], np.int8), "hol": np.zeros(2, np.int8), "dtype": np.zeros(2, np.int8),
            "is_missing": np.zeros((2, 96), bool)}


def toy_model():
    model = hmm.CondHMM(K=3, dz=14, cond=False, ar=True, seed=0).double()
    with torch.no_grad():
        model.rho.fill_(-30.0)
        model.rho[0].zero_()
        model.s.fill_(math.log(math.e ** 2 - 1))        # sigma = 1 + softplus(s) = 3
        model.psi.fill_(math.log(4.0))                   # phi = 0.8
    return model


def test_e0_stationary_when_last_obs_masked():
    model, m, C = toy_model(), np.zeros((2, 96)), np.array([1e9, 1e9, 1e9])
    masked = toy_panel()
    masked["Y"][0] = np.nan
    out = hmm.forecast(model, masked, 1, m, C, N=20000, seed=0, device="cpu")
    assert np.var(out["paths"][:, 0]) == pytest.approx(25.0, rel=0.10)    # sigma^2 / (1 - phi^2)
    out = hmm.forecast(model, toy_panel(), 1, m, C, N=20000, seed=0, device="cpu")
    assert np.var(out["paths"][:, 0]) == pytest.approx(9.0, rel=0.10)     # e0 = 0 observed


def test_analytic_matches_mc_when_phi_zero():
    p = ft.load_panel()
    model = rand_model(K=3, dz=14, cond=True, ar=False, seed=5, scale=0.3).float()
    m = np.tile(np.linspace(60, 170, 96), (257, 1))
    d = p["dates"].index(date(2021, 7, 7))
    C = np.array([185.0, 195.0, 205.0])
    mc = (hmm.forecast(model, p, d, m, C, N=20000, seed=0, device="cpu")["paths"].max(1)[:, None] > C).mean(0)
    an = hmm.analytic_risk(model, p, d, m, C, device="cpu")
    tol = 3 * np.sqrt(an * (1 - an) / 20000) + 1 / 20000
    assert (np.abs(mc - an) <= tol).all()


def test_sigma_re():
    assert hmm.sigma_re(np.array([3.0]), np.array([0.8]), 4.0)[0] == pytest.approx(1.8)
    assert hmm.sigma_re(np.array([1.0]), np.array([0.0]), 50.0)[0] == pytest.approx(1.0)


def test_re_adds_day_level_variance(fitted):
    p, _, state = fitted
    d = p["dates"].index(date(2021, 7, 7))
    base = hmm.hmm_model("B4", tau=10.0, half_life=60, N=2000, device="cpu")
    re = hmm.hmm_model("B4", tau=10.0, half_life=60, N=2000, re=True, device="cpu")
    assert state["s_op"][1] > 0
    assert np.var(re["predict"](state, p, d)["paths"].mean(1)) > np.var(base["predict"](state, p, d)["paths"].mean(1))


def test_final_epochs():
    assert hmm.final_epochs([10, 13, 20, 8]) == 12 and hmm.final_epochs([10, 15]) == 13 and hmm.final_epochs([7]) == 7


def test_inner_state_and_test_fold_rules(fitted):
    p, model, state = fitted
    assert state["best_epoch"] >= 1 and np.isfinite(state["history"][0]["val_nll"])
    assert not all(torch.equal(x, y) for x, y in zip(state["model"].state_dict().values(),
                                                     state["inner_state"]["model"].state_dict().values()))
    with pytest.raises(AssertionError):
        model["fit"](p, "test", np.array([186.5, 198.0, 210.7]))
    with pytest.raises(AssertionError):
        hmm.hmm_model("B4", protocol="B", train_from=date(2021, 7, 1), max_epochs=1, device="cpu")["fit"](
            p, "f1", np.array([181.0, 189.5, 198.0]))


@pytest.mark.parametrize("kind", ["B2", "B3", "B4"])
def test_ladder_rows(kind):
    p = ft.load_panel()
    slots, days, states, inner = ev.rolling_origin(
        hmm.hmm_model(kind, tau=10.0, half_life=60, max_epochs=2, N=50, device="cpu"), p)
    assert slots.height == 5376 and days.height == 56 and inner.height == 28
    assert set(days["model"].unique()) == {kind}
    assert slots["state_smooth"].null_count() == 0 and set(slots["state_smooth"].unique()) <= {1, 2, 3}
    assert all(s["best_epoch"] >= 1 for s in states.values())
```

- [ ] **Step 2: Run to verify they fail** — `uv run pytest tests/test_hmm.py -q` → `AttributeError: module 'gmst.hmm' has no attribute 'hmm_model'`.
- [ ] **Step 3: Implement** the Task 12 functions.
- [ ] **Step 4: Run to verify they pass** — `uv run pytest tests/test_hmm.py -q` → `32 passed`.
- [ ] **Step 5: Full suite and commit**
```bash
git add gmst/hmm.py tests/test_hmm.py
git commit -m "feat: MC peak risk with masked-e0 draw, analytic phi=0 risk, B2/B3/B4 factory with inner early stopping + refit, re diagnostic"
```

---

### Task 13: P0-4 leakage test (post-origin perturbation) for every model

**Files:**
- Create: `tests/test_leakage.py` (tests only; fix any model code it exposes in the module that leaks, and list those fixes in the commit message)

**Interfaces:**
- Consumes: `bl.b0_model`, `bl.b0p_model`, `bb.bb_model`, `bl.b1_model`, `hmm.hmm_model`, `ev.thresholds`, `ev.calibrate_oof`, `ft.*`.

**What it proves (v2 §15.7-1, FR-33, US-011):** at origin `d0 = 2021-08-18` (first f4 validation day), predictions of B0, B0′, BB, B1, B2, B3, B4 are bitwise identical when `Y[d0:]`, weather `X[d0:]` and production `X[d0:]` (A+ does not allow same-day production or weather) are replaced by random numbers. SCENARIO (B4 + protocol B) is identical when production from `d0+1` and everything else after the origin change, and **differs** when production of `d0` changes. B1 with `A+W*` differs when weather of `d0` changes. Fits (thresholds, backbone profiles, B1, HMM with inner early stopping) are identical when every value after the fold's last train day changes. Calibration of fold j ignores fold j labels. The default panel is sealed.

- [ ] **Step 1: Write the test** — create `tests/test_leakage.py`:

```python
from datetime import date

import numpy as np
import polars as pl
import pytest
import torch

from gmst import backbone as bb
from gmst import baselines as bl
from gmst import evaluate as ev
from gmst import features as ft
from gmst import hmm

D0 = date(2021, 8, 18)
WX = ("기온", "풍속", "습도", "강수량_증분")
HMM_KW = dict(tau=10.0, half_life=60, max_epochs=3, N=100, device="cpu")
MAIN = {
    "B0": lambda: bl.b0_model(),
    "B0p": lambda: bl.b0p_model(),
    "BB": lambda: bb.bb_model(10.0, 60),
    "B1": lambda: bl.b1_model(10.0, 60, rounds=20),
    "B2": lambda: hmm.hmm_model("B2", **HMM_KW),
    "B3": lambda: hmm.hmm_model("B3", **HMM_KW),
    "B4": lambda: hmm.hmm_model("B4", **HMM_KW),
}


@pytest.fixture(scope="module")
def panel():
    return ft.load_panel()


def perturb(panel, y_from, wx_from, prod_from, seed=1):
    rng = np.random.default_rng(seed)
    p = {**panel, "Y": panel["Y"].copy(), "X": {k: v.copy() for k, v in panel["X"].items()}}
    p["Y"][y_from:] = rng.uniform(0, 250, p["Y"][y_from:].shape)
    for k in WX:
        p["X"][k][wx_from:] = rng.uniform(0, 40, p["X"][k][wx_from:].shape)
    p["X"]["생산량"][prod_from:] = rng.uniform(0, 3000, p["X"]["생산량"][prod_from:].shape)
    return p


def same(a, b):
    for k in ("y_mean", "y_median", "q", "paths", "risk_raw"):
        x, y = a.get(k), b.get(k)
        assert (x is None) == (y is None), k
        if x is not None:
            assert np.array_equal(np.asarray(x), np.asarray(y)), k
    assert a["M_hat_median"] == b["M_hat_median"] and a["M_hat_mean"] == b["M_hat_mean"]
    assert a["peak_time_mode"] == b["peak_time_mode"]


@pytest.fixture(scope="module")
def fitted(panel):
    C, _ = ev.thresholds(panel, ft.role_idx(panel, "f4", "train"))
    models = {k: f() for k, f in MAIN.items()}
    models["SCN"] = hmm.hmm_model("B4", protocol="B", train_from=date(2021, 7, 1), **HMM_KW)
    return {k: (m, m["fit"](panel, "f4", C)) for k, m in models.items()}


@pytest.mark.parametrize("name", list(MAIN))
def test_main_track_ignores_post_origin_values(panel, fitted, name):
    model, state = fitted[name]
    d = panel["dates"].index(D0)
    same(model["predict"](state, panel, d), model["predict"](state, perturb(panel, d, d, d), d))


def test_scenario_track_uses_only_same_day_production(panel, fitted):
    model, state = fitted["SCN"]
    d = panel["dates"].index(D0)
    base = model["predict"](state, panel, d)
    same(base, model["predict"](state, perturb(panel, d, d, d + 1), d))
    p = {**panel, "X": {k: v.copy() for k, v in panel["X"].items()}}
    p["X"]["생산량"][d] = p["X"]["생산량"][d][::-1] * 3 + 500.0
    assert not np.array_equal(base["paths"], model["predict"](state, p, d)["paths"])


def test_awstar_uses_same_day_weather_only(panel):
    C, _ = ev.thresholds(panel, ft.role_idx(panel, "f4", "train"))
    m = bl.b1_model(10.0, 60, protocol="A+W*")
    st = m["fit"](panel, "f4", C)
    d = panel["dates"].index(D0)
    base = m["predict"](st, panel, d)
    same(base, m["predict"](st, perturb(panel, d, d + 1, d), d))
    p = {**panel, "X": {k: v.copy() for k, v in panel["X"].items()}}
    for k in WX:
        p["X"][k][d] = p["X"][k][d] + 25.0
    assert not np.array_equal(base["y_mean"], m["predict"](st, p, d)["y_mean"])


def test_fits_ignore_values_after_train_end(panel):
    tr = ft.role_idx(panel, "f4", "train")
    after = int(tr.max()) + 1                               # the gap day 08-17 onward
    p2 = perturb(panel, after, after, after)
    C1, _ = ev.thresholds(panel, tr)
    assert np.array_equal(C1, ev.thresholds(p2, tr)[0])
    a, _ = bb.fit_backbone(panel["Y"], panel["op"], panel["dtype"], tr, 10.0, 60)
    b, _ = bb.fit_backbone(p2["Y"], p2["op"], p2["dtype"], tr, 10.0, 60)
    assert np.array_equal(a, b)
    d = panel["dates"].index(date(2021, 8, 10))
    for make in (lambda: bl.b1_model(10.0, 60, rounds=20), lambda: hmm.hmm_model("B4", **HMM_KW)):
        m = make()
        s1, s2 = m["fit"](panel, "f4", C1), m["fit"](p2, "f4", C1)
        same(m["predict"](s1, panel, d), m["predict"](s2, panel, d))
        if isinstance(s1.get("model"), torch.nn.Module):
            assert all(torch.equal(x, y) for x, y in zip(s1["model"].state_dict().values(),
                                                         s2["model"].state_dict().values()))


def test_calibration_ignores_own_fold_labels():
    rng = np.random.default_rng(0)
    raw = rng.random(160)
    evt = (rng.random(160) < raw).astype(float)
    folds = np.repeat(["f1", "f2", "f3", "f4"], 40)
    mk = lambda e: pl.DataFrame({"fold": folds, "variant": ["main"] * 160, "model": ["M"] * 160,
                                 "usable_peak": [True] * 160,
                                 **{f"risk_raw_C{c}": raw for c in (50, 75, 90)},
                                 **{f"risk_{m}_C{c}": np.full(160, np.nan) for m in ("platt", "iso") for c in (50, 75, 90)},
                                 **{f"event_C{c}": e for c in (50, 75, 90)}})
    flipped = evt.copy()
    flipped[folds == "f3"] = 1 - flipped[folds == "f3"]
    a, _ = ev.calibrate_oof(mk(evt), "platt")
    b, _ = ev.calibrate_oof(mk(flipped), "platt")
    f3 = lambda d: d.filter(pl.col("fold") == "f3")["risk_platt_C75"].to_numpy()
    assert np.array_equal(f3(a), f3(b))


def test_sealed_by_default(panel):
    t = ft.role_idx(panel, "test", "test")
    assert np.isnan(panel["Y"][t]).all() and all(np.isnan(v[t]).all() for v in panel["X"].values())
```

- [ ] **Step 2: Run it** — `uv run pytest tests/test_leakage.py -q`. Expected: `12 passed`. Any failure is a leak in the model it names: fix the model module (not the test) until it passes; rerun Tasks 6–12 tests.
- [ ] **Step 3: Full suite (GPU and CPU)** — `uv run pytest -q` and `CUDA_VISIBLE_DEVICES="" uv run pytest -q`.
- [ ] **Step 4: Commit**
```bash
git add tests/test_leakage.py
git commit -m "test: P0-4 leakage test (post-origin perturbation, fit invariance, calibration, seal) for B0..B4 and SCENARIO"
```
(also `git add` any model module fixed in Step 2).

---

### Task 14: P1-3 three-seed state stability and the transition table

**Files:**
- Modify: `gmst/hmm.py` (append)
- Create: `tests/test_state_stability.py`

**Interfaces:**
- Produces (module `gmst.hmm`):
  - `transition_table(model, panel, day_idx, protocol="A+") -> pl.DataFrame` — columns `op, daytype, hour, from_state, p_to_high`; one row per (op, dtype) cell present among `day_idx` (op = `cal_flags` op, sorted), hour 0–23, from_state 1…K; `p_to_high` = mean over the hour's 4 slots of `A_t(from_state, K)` with `z` built for that cell (`hol = 0`; protocol B production features 0); `daytype` as `wk/sat/sun` (FR-93, v2 §13.4).
  - `conclusion(table, K) -> tuple[int, str, int]` — the `(op, daytype, hour)` row with the largest `p_to_high` among rows with `from_state == K − 1` (entry into the high state; first in table order on ties) (**[plan pin]** of "§13.4 결론").
  - `state_summary(model, panel, train_idx, eval_idx, m, protocol="A+") -> dict` — `delta` ((K, 2) numpy), `occ` ((K,) mean smoothed occupancy over observed target steps of the train blocks), `conclusion` (from `transition_table` over `train_idx`), `eval_states` (1-based smoothed argmax on observed slots of `eval_idx` days, concatenated in day order).
  - `stability_rows(summaries: dict[int, dict]) -> list[dict]` — for each seed pair `(a, b)` in insertion order: `seed_a, seed_b, occ_maxdiff, delta_maxdiff, same_order` (δ strictly increasing in k for both op columns in both seeds), `same_conclusion`, `agreement` (share of equal `eval_states`).
  - `state_stability(panel, fold="f1", seeds=(0, 1, 2), m=None, tau=10.0, half_life=60, protocol="A+", **train_kw) -> dict` — `tr = role_idx(panel, fold, "train")`, `inn = inner_idx ∩ tr`; `m_full`/`m_in` = `m` if given else `fold_backbone` on `tr` / `tr \ inn`; per seed `fit_with_inner(..., seed=seed, **train_kw)` (the same R10 procedure as B4) and `state_summary(model, panel, tr, role_idx(panel, fold, "val"), m_full, protocol)`. Returns `{"rows": stability_rows(...), "per_seed": {seed: summary}, "ok": all(occ_maxdiff < 0.05 and delta_maxdiff < 5 and same_order and same_conclusion), "collapsed": any(min(occ) < 0.02)}` (v2 §15.7-2, A31). The real-data verdict is **reported** (`results/state_stability.csv`, "seed 의존" when not ok) — it is not a pass/fail gate; the gate is the synthetic test below. If `collapsed`, Task 20 reruns with `--occ-floor`.

- [ ] **Step 1: Write the failing test** — create `tests/test_state_stability.py`:

```python
from datetime import date, timedelta

import numpy as np
import polars as pl

from gmst import features as ft
from gmst import hmm


def synthetic_panel(n=40, seed=0):
    rng = np.random.default_rng(seed)
    level = np.r_[np.full(32, -20.0), np.zeros(32), np.full(32, 20.0)]   # low 00-08h, normal 08-16h, high 16-24h
    dates = [date(2021, 3, 1) + timedelta(days=i) for i in range(n)]
    roles = ["train"] * (n - 8) + ["gap"] + ["val"] * 7
    return {"dates": dates, "Y": level + rng.normal(0, 2, (n, 96)),
            "X": {k: np.zeros((n, 96)) for k in ("생산량", "기온", "풍속", "습도", "강수량_증분")},
            "is_missing": np.zeros((n, 96), bool), "op": np.ones(n, np.int8), "hol": np.zeros(n, np.int8),
            "dtype": np.zeros(n, np.int8), "dow": np.zeros(n, np.int8), "month": np.full(n, 3, np.int8),
            "days": pl.DataFrame({"date": dates, "f1": roles})}


def test_transition_table_schema():
    p = synthetic_panel()
    t = hmm.transition_table(hmm.CondHMM(K=3, dz=14, seed=0), p, np.arange(40))
    assert t.columns == ["op", "daytype", "hour", "from_state", "p_to_high"]
    assert t.height == 24 * 3 and set(t["daytype"].unique()) == {"wk"}
    assert ((t["p_to_high"] >= 0) & (t["p_to_high"] <= 1)).all()


def test_stability_rows_arithmetic():
    s = {0: {"delta": np.array([[-2.0, -20.0], [0.0, 0.0], [2.0, 20.0]]), "occ": np.array([0.3, 0.4, 0.3]),
             "conclusion": (1, "wk", 16), "eval_states": np.array([1, 2, 3, 3])},
         1: {"delta": np.array([[-2.0, -21.0], [0.0, 1.0], [2.0, 20.5]]), "occ": np.array([0.32, 0.4, 0.28]),
             "conclusion": (1, "wk", 16), "eval_states": np.array([1, 2, 2, 3])}}
    (r,) = hmm.stability_rows(s)
    assert (r["seed_a"], r["seed_b"]) == (0, 1)
    assert abs(r["occ_maxdiff"] - 0.02) < 1e-12
    assert abs(r["delta_maxdiff"] - 1.0) < 1e-12 and r["same_order"] and r["same_conclusion"]
    assert r["agreement"] == 0.75


def test_synthetic_three_states_are_seed_stable():
    p = synthetic_panel()
    out = hmm.state_stability(p, fold="f1", seeds=(0, 1, 2), m=np.zeros((40, 96)), max_epochs=200, device="cpu")
    assert [(r["seed_a"], r["seed_b"]) for r in out["rows"]] == [(0, 1), (0, 2), (1, 2)]
    for r in out["rows"]:
        assert r["same_order"] and r["same_conclusion"]
        assert r["occ_maxdiff"] < 0.05 and r["delta_maxdiff"] < 5 and r["agreement"] > 0.95
    assert all(s["conclusion"] == (1, "wk", 16) for s in out["per_seed"].values())
    assert out["ok"] is True and out["collapsed"] is False


def test_real_f1_report_structure():
    p = ft.load_panel()
    out = hmm.state_stability(p, fold="f1", seeds=(0, 1, 2), max_epochs=3, device="cpu")
    assert [(r["seed_a"], r["seed_b"]) for r in out["rows"]] == [(0, 1), (0, 2), (1, 2)]
    for r in out["rows"]:
        assert r["same_order"] is True and 0 <= r["agreement"] <= 1
        assert np.isfinite(r["occ_maxdiff"]) and np.isfinite(r["delta_maxdiff"])
    assert out["ok"] == all(r["occ_maxdiff"] < 0.05 and r["delta_maxdiff"] < 5 and r["same_conclusion"]
                            for r in out["rows"])
    assert isinstance(out["collapsed"], bool)
```

- [ ] **Step 2: Run to verify it fails** — `uv run pytest tests/test_state_stability.py -q` → `AttributeError: module 'gmst.hmm' has no attribute 'transition_table'`.
- [ ] **Step 3: Implement** the Task 14 functions.
- [ ] **Step 4: Run to verify it passes** — `uv run pytest tests/test_state_stability.py -q` → `4 passed` (the synthetic gate trains 3 seeds × 2 fits; expect well under 2 minutes on CPU).
- [ ] **Step 5: Full suite and commit**
```bash
git add gmst/hmm.py tests/test_state_stability.py
git commit -m "feat: transition table, 3-seed state stability (synthetic gate + real-data report)"
```

---

### Task 15: KEPCO 산업용(을) 2021 tariff calculator and ratchet floor

**Files:**
- Create: `gmst/scenario.py` (tariff part), `tests/test_scenario.py`

**Interfaces:**
- Consumes: `splits.HOLIDAYS_2021`, `features.load_panel` (for the floor test).
- Produces (module `gmst.scenario`):
  - `TARIFF = {"I": {"base": 7220, "summer": (61.6, 114.5, 196.6), "springfall": (61.6, 84.1, 114.8), "winter": (68.6, 114.7, 172.2)}, "II": {"base": 8320, "summer": (56.1, 109.0, 191.1), "springfall": (56.1, 78.6, 109.3), "winter": (63.1, 109.2, 166.7)}, "III": {"base": 9810, "summer": (55.2, 108.4, 178.7), "springfall": (55.2, 77.3, 101.0), "winter": (62.5, 108.6, 155.5)}}` — tuples are (off, mid, peak) ₩/kWh; base ₩/kW (FR-82, `.sdd/kepco_tariff_2021.md`).
  - `RATCHET_MONTHS = (12, 1, 2, 7, 8, 9)`
  - `season_name(month) -> str` — `summer` 6–8, `springfall` 3–5 and 9–10, `winter` 11–2.
  - `tariff_holidays(tariff_0816=True) -> set[date]` — `set(HOLIDAYS_2021)`, minus 2021-08-16 when `False` (FR-83, ruling Q1).
  - `band(ts, holiday, for_energy) -> int` — 0 off, 1 mid, 2 peak. `holiday` or Sunday (`ts.weekday() == 6`, ruling R1) → 0. Summer/spring-fall: off `h < 9 or h ≥ 23`; peak `10 ≤ h < 12 or 13 ≤ h < 17`; else mid. Winter: off `h < 9 or h ≥ 23`; peak `10 ≤ h < 12 or 17 ≤ h < 20 or 22 ≤ h < 23`; else mid (`h = hour + minute/60`). Saturday (non-holiday) peak → 1 when `for_energy` (demand keeps 2).
  - `rate(ts, holiday, option="II") -> float` — `TARIFF[option][season_name(ts.month)][band(ts, holiday, True)]`.
  - `energy_won(y, ts, holidays, option="II") -> float` — `Σ 0.25 · y · rate` over finite `y`, `holiday = ts.date() in holidays` (FR-84).
  - `billing_demand(y, ts, holidays) -> float` — max `y` over slots with `band(ts, holiday, for_energy=False) ≥ 1`; 0.0 when none.
  - `ratchet_floor(panel, month, exclude_dates, holidays) -> tuple[float, datetime]` — max finite `panel["Y"]` (loader masks apply, so masked points are excluded) over days whose month is in `{m ∈ RATCHET_MONTHS : m ≤ month} ∪ {month}` and not in `exclude_dates`, not Sunday, not in `holidays`, slots with demand band ≥ 1; returns the value and its interval-start timestamp. Marker: `# ponytail: 2020-12 자료 없음·계약전력 30% 하한 미적용·마스킹 점 제외로 래칫 바닥 계산, 실제 청구 이력이 오면 교체 (FR-85)`.
  - `expected_p_app(month_max, floor) -> float` = `mean(max(month_max, floor))` over paths (FR-85, A29).
  - `basic_won(p_app, option="II") -> float` = `TARIFF[option]["base"] · p_app`.
- The string `전기요금` must not appear anywhere in `scenario.py` (DC7).

- [ ] **Step 1: Write the failing test** — create `tests/test_scenario.py`:

```python
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pytest

from gmst import features as ft
from gmst import scenario as sc
from gmst import splits as sp

H = set(sp.HOLIDAYS_2021)


@pytest.mark.parametrize("ts,won", [
    (datetime(2021, 7, 7, 10, 0), 4777.5), (datetime(2021, 7, 10, 11, 0), 2725.0),
    (datetime(2021, 7, 11, 11, 0), 1402.5), (datetime(2021, 8, 16, 11, 0), 1402.5),
    (datetime(2021, 1, 5, 18, 0), 4167.5), (datetime(2021, 2, 13, 11, 0), 1577.5),
    (datetime(2021, 9, 8, 10, 0), 2732.5)])
def test_energy_cases(ts, won):
    assert sc.energy_won(np.array([100.0]), [ts], H) == pytest.approx(won)


@pytest.mark.parametrize("ts,band", [
    (datetime(2021, 7, 7, 9, 0), 1), (datetime(2021, 7, 7, 8, 45), 0), (datetime(2021, 7, 7, 17, 0), 1),
    (datetime(2021, 7, 7, 23, 0), 0), (datetime(2021, 1, 5, 13, 0), 1), (datetime(2021, 1, 5, 21, 0), 1),
    (datetime(2021, 1, 5, 22, 30), 2), (datetime(2021, 7, 10, 11, 0), 2)])
def test_bands(ts, band):
    assert sc.band(ts, holiday=False, for_energy=False) == band


def test_saturday_sunday_holiday_rules():
    assert sc.band(datetime(2021, 7, 10, 11, 0), holiday=False, for_energy=True) == 1
    assert sc.band(datetime(2021, 7, 7, 11, 0), holiday=True, for_energy=False) == 0
    assert sc.band(datetime(2021, 7, 11, 11, 0), holiday=False, for_energy=False) == 0


def test_tariff_0816_switch():
    assert sc.tariff_holidays(True) == H and sc.tariff_holidays(False) == H - {date(2021, 8, 16)}
    assert sc.energy_won(np.array([100.0]), [datetime(2021, 8, 16, 11, 0)], sc.tariff_holidays(False)) == pytest.approx(4777.5)


def test_tariff_table():
    assert {k: v["base"] for k, v in sc.TARIFF.items()} == {"I": 7220, "II": 8320, "III": 9810}
    assert sc.TARIFF["II"]["summer"] == (56.1, 109.0, 191.1)
    assert sc.TARIFF["II"]["springfall"] == (56.1, 78.6, 109.3)
    assert sc.TARIFF["II"]["winter"] == (63.1, 109.2, 166.7)
    assert sc.TARIFF["I"]["winter"] == (68.6, 114.7, 172.2) and sc.TARIFF["III"]["summer"] == (55.2, 108.4, 178.7)
    assert [sc.season_name(m) for m in (1, 3, 6, 9, 10, 11)] == [
        "winter", "springfall", "summer", "springfall", "springfall", "winter"]


def test_billing_demand_rules():
    day = [datetime(2021, 7, 7, h, mi) for h in range(24) for mi in (0, 15, 30, 45)]
    y = np.where([t.hour < 9 or t.hour >= 23 for t in day], 200.0, 150.0)
    assert sc.billing_demand(y, day, H) == 150.0
    hol = [datetime(2021, 8, 16, h, mi) for h in range(24) for mi in (0, 15, 30, 45)]
    assert sc.billing_demand(np.full(96, 300.0), hol, H) == 0.0
    assert sc.billing_demand(np.array([180.0]), [datetime(2021, 7, 10, 11, 0)], H) == 180.0


def test_floor_222():
    p = ft.load_panel()
    u = ft.usable_peak(p)
    tgt = lambda folds: {p["dates"][i] for f in folds for i in ft.role_idx(p, f, "val") if u[i] and p["op"][i] == 1}
    at = datetime(2021, 7, 19, 11, 15)
    assert sc.ratchet_floor(p, 7, tgt(["f2"]), H) == (222.0, at)
    assert sc.ratchet_floor(p, 8, tgt(["f2", "f3", "f4"]), H) == (222.0, at)


def test_basic_charge_delta():
    assert sc.basic_won(10.0) == pytest.approx(83200.0)
    assert sc.expected_p_app(np.array([200.0, 230.0, 240.0]), 222.0) == pytest.approx((222 + 230 + 240) / 3)


def test_no_data_tariff():
    assert "전기요금" not in Path(sc.__file__).read_text(encoding="utf-8")
```

- [ ] **Step 2: Run to verify it fails** — `uv run pytest tests/test_scenario.py -q` → `ModuleNotFoundError: No module named 'gmst.scenario'`.
- [ ] **Step 3: Implement** the tariff part of `gmst/scenario.py`.
- [ ] **Step 4: Run to verify it passes** — `uv run pytest tests/test_scenario.py -q` → `22 passed`.
- [ ] **Step 5: Full suite and commit**
```bash
git add gmst/scenario.py tests/test_scenario.py
git commit -m "feat: KEPCO industrial(eul) 2021 TOU tariff, Sunday/holiday/Saturday metering, ratchet floor 222"
```

---

### Task 16: SCENARIO track and production-schedule what-ifs

**Files:**
- Modify: `gmst/scenario.py` (append), `tests/test_scenario.py` (append)

**Interfaces:**
- Consumes: `hmm.hmm_model` (protocol `B`, `train_from=date(2021, 7, 1)`), `hmm.forecast`, Task 15 tariff functions, `features.*`, `evaluate.thresholds`.
- Produces (module `gmst.scenario`):
  - `TRANSFORMS = ("shift_peak", "stagger_start", "avoid_high", "ease_peak")`, `RATES = (0.1, 0.2, 0.3)`
  - `shift_peak(P, day, s, holidays) -> np.ndarray`, `stagger_start(P, s) -> np.ndarray`, `avoid_high(P, s, y_med, c90) -> np.ndarray`, `ease_peak(P, s, peak_slot) -> np.ndarray` — `P` = 24 hourly production values of one day; all return a new array with the same sum (1e-9), non-negative; unchanged when there is no donor or no receiver (FR-88).
  - `c_change(P0, Pa) -> float` = `Σ|Pa − P0| / (2 ΣP0)` (0.0 if `ΣP0 == 0`).
  - `SCN_DAY_COLS = ["fold", "date", "scenario", "rate", "E_M", "M_median", "risk_C50", "risk_C75", "risk_C90", "risk_ok", "peak_time_mode", "C_change", "d_energy_won", "eta_star"]`
  - `SCN_MONTH_COLS = ["month", "scenario", "rate", "n_days", "P_floor", "E_P_app_base", "E_P_app_scn", "d_demand_won", "d_energy_won", "d_kwh", "d_total_won", "d_total_pct", "C_change", "eta_star", "risk_ok_all"]`
  - `run_scenarios(panel, states, N=2000, holidays=None, max_days=None) -> tuple[pl.DataFrame, pl.DataFrame]` — `states` = {fold: SCENARIO state} for f2–f4 (from `rolling_origin` of `hmm_model("B4", protocol="B", train_from=date(2021, 7, 1))`); `holidays` default `tariff_holidays(True)`.

**Algorithm (v2 §12, §12.1, FR-85…FR-89, A24, A28, A29, rulings R7, R8):**
- Transforms (hours h = 0…23, `s` = rate):
  - `shift_peak`: donors = hours with `band(datetime(day, h), holiday=day in holidays, for_energy=False) == 2` and `P > 0` (the demand-setting peak band; none on Sundays/holidays); receivers = other hours with `P > 0`; move `s·P[donor]` from each donor and spread the total over receivers proportionally to `P[receiver]`.
  - `stagger_start`: for each maximal run of consecutive hours with `P > 0` and length ≥ 2, move `s·P[first]` to the run's second hour.
  - `avoid_high`: donors = hours with `P > 0` whose 4 slots contain any `y_med > c90` (baseline median path vs C90); receivers = other hours with `P > 0`; proportional spread.
  - `ease_peak`: donors = hours covering slots `peak_slot − 2 … peak_slot + 2` (clipped to 0–95) with `P > 0`; receivers = other hours with `P > 0`; proportional spread.
- `run_scenarios`: for each fold in `states` (sorted) the target days are that fold's validation days with `op == 1` and `usable_peak` (first `max_days` if given). For each target day `d`: `P0 = X["생산량"][d, ::4]`; `base = forecast(state["model"], panel, d, state["m"], state["C"], "B", N, device=state["device"])` (date seed); scenarios = `("baseline", 0.0)` then every transform × rate. For each scenario build `Pa` (baseline: `P0`; `avoid_high` uses `base["y_median"]` and `state["C"][2]`; `ease_peak` uses `base["peak_time_mode"]`), a shallow panel copy with `X["생산량"][d] = np.repeat(Pa, 4)` (copy only that array), forecast with the **same seed** (common random numbers). Daily row: `E_M = M_hat_mean`, `M_median = M_hat_median`, `risk_C* = risk_raw` (raw MC risk; Platt is monotone so `risk_ok` is unaffected — **[plan pin]**), `risk_ok = risk_C90 ≤ base risk_C90`, `peak_time_mode`, `C_change = c_change(P0, Pa)`, `d_energy_won = energy_won(y_mean_a) − energy_won(y_mean_0)` for that day, `eta_star = −d_energy_won / C_change` (NaN when `C_change == 0`) — the daily J_d(a) components (v2 §12, R7; `eta_star` in the daily file follows v2 §12, see Spec issues). `date` as `%Y.%m.%d`, `rate` Float64.
- Monthly rows (per calendar month of the target days × scenario): `P_floor = ratchet_floor(panel, month, exclude_dates = that month's target days, holidays)[0]`; per path `n`, `M_mp^(n) = max over the month's target days (skip Sunday/holiday days) of max over demand-band (≥ 1) slots of the path` — dates paired by path index, marker `# ponytail: 날짜 간 경로 독립 짝짓기, 날짜 간 상관이 크면 일 랜덤효과 경로로 교체 (FR-85)`; `E_P_app_base/scn = expected_p_app(M_mp, P_floor)` (all −inf → `P_floor`); `d_demand_won = basic_won(E_P_app_scn − E_P_app_base)`; `d_energy_won` = sum of daily; `d_kwh = Σ_d Σ_q 0.25 (y_mean_a − y_mean_0)` (R8); `d_total_won = d_demand_won + d_energy_won`; `d_total_pct = 100 · d_total_won / (basic_won(E_P_app_base) + Σ_d energy_won(y_mean_0))`; `C_change = Σ_d Σ_h |ΔP| / (2 Σ_d Σ_h P0)`; `eta_star = −d_total_won / C_change` (NaN if 0); `risk_ok_all = all(risk_ok)`; `n_days`. With the floor at 222 (07-19 11:15) `d_demand_won` is ≈ 0 in July/August unless a path's monthly max exceeds 222; the energy term is the main lever (ruling).

- [ ] **Step 1: Write the failing tests** — append to `tests/test_scenario.py`:

```python
import polars as pl

from gmst import evaluate as ev
from gmst import hmm

P0 = np.array([0.0] * 8 + [10.0] * 10 + [0.0] * 6)       # production 08:00-17:59
DAY = date(2021, 7, 7)                                      # Wednesday, summer


def test_shift_peak():
    P = sc.shift_peak(P0, DAY, 0.1, H)
    assert P.sum() == pytest.approx(P0.sum(), abs=1e-9) and (P >= 0).all()
    assert P[[8, 9, 12, 17]].tolist() == pytest.approx([11.5] * 4)
    assert P[[10, 11, 13, 14, 15, 16]].tolist() == pytest.approx([9.0] * 6)
    assert sc.c_change(P0, P) == pytest.approx(0.06)
    assert np.array_equal(sc.shift_peak(P0, date(2021, 7, 11), 0.1, H), P0)       # Sunday: no peak band


def test_stagger_start():
    P = sc.stagger_start(P0, 0.2)
    assert (P[8], P[9]) == pytest.approx((8.0, 12.0)) and P.sum() == pytest.approx(100.0)
    single = np.r_[5.0, np.zeros(23)]
    assert np.array_equal(sc.stagger_start(single, 0.2), single)


def test_avoid_high():
    y_med = np.full(96, 100.0)
    y_med[45] = 250.0                                           # slot 45 = hour 11 above C90
    P = sc.avoid_high(P0, 0.3, y_med, 210.7)
    assert P[11] == pytest.approx(7.0)
    assert P[[8, 9, 10, 12, 13, 14, 15, 16, 17]].tolist() == pytest.approx([10 + 1 / 3] * 9)
    assert np.array_equal(sc.avoid_high(P0, 0.3, np.full(96, 100.0), 210.7), P0)


def test_ease_peak():
    P = sc.ease_peak(P0, 0.1, 58)                               # slots 56..60 -> hours 14, 15
    assert P[[14, 15]].tolist() == pytest.approx([9.0, 9.0])
    assert P[[8, 9, 10, 11, 12, 13, 16, 17]].tolist() == pytest.approx([10.25] * 8)


def test_c_change_edges():
    assert sc.c_change(P0, P0) == 0.0 and sc.c_change(np.zeros(24), np.zeros(24)) == 0.0


def test_scenario_state_train_days():
    p = ft.load_panel()
    m = hmm.hmm_model("B4", protocol="B", train_from=date(2021, 7, 1), max_epochs=1, N=20, device="cpu")
    for f, n in (("f2", 19), ("f4", 47)):
        st = m["fit"](p, f, ev.thresholds(p, ft.role_idx(p, f, "train"))[0])
        ds = [p["dates"][i] for i in st["train_idx"]]
        assert min(ds) == date(2021, 7, 1) and len(ds) == n


def same_frame(a, b):
    assert a.columns == b.columns
    for c in a.columns:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        assert (np.array_equal(x, y, equal_nan=True) if x.dtype.kind == "f" else (x == y).all()), c


def test_run_scenarios_small():
    p = ft.load_panel()
    model = hmm.hmm_model("B4", protocol="B", train_from=date(2021, 7, 1), max_epochs=2, N=100, device="cpu")
    states = {"f4": model["fit"](p, "f4", ev.thresholds(p, ft.role_idx(p, "f4", "train"))[0])}
    days, month = sc.run_scenarios(p, states, N=100, max_days=2)
    assert days.columns == sc.SCN_DAY_COLS and month.columns == sc.SCN_MONTH_COLS
    assert days.height == 2 * 13 and month.height == 13
    base = days.filter(pl.col("scenario") == "baseline")
    assert (base["d_energy_won"] == 0).all() and (base["C_change"] == 0).all() and base["risk_ok"].all()
    mb = month.filter(pl.col("scenario") == "baseline")
    assert (mb["d_demand_won"] == 0).all() and (mb["d_kwh"] == 0).all() and (month["P_floor"] == 222.0).all()
    shifted = days.filter(pl.col("scenario") != "baseline")
    assert (shifted["C_change"] > 0).any()
    again, _ = sc.run_scenarios(p, states, N=100, max_days=2)
    same_frame(days, again)
```

- [ ] **Step 2: Run to verify they fail** — `uv run pytest tests/test_scenario.py -q` → `AttributeError: module 'gmst.scenario' has no attribute 'shift_peak'`.
- [ ] **Step 3: Implement** the Task 16 functions.
- [ ] **Step 4: Run to verify they pass** — `uv run pytest tests/test_scenario.py -q` → `29 passed`.
- [ ] **Step 5: Full suite and commit**
```bash
git add gmst/scenario.py tests/test_scenario.py
git commit -m "feat: SCENARIO what-ifs (4 transforms x 3 rates, common random numbers), daily J_d components, monthly ratchet/energy KRW with kWh change"
```

---

### Task 17: Error analysis, FN/FP conditions, augmentation-leakage gap, transition table file

**Files:**
- Create: `gmst/analysis.py`, `tests/test_analysis.py`

**Interfaces:**
- Consumes: `evaluate` (OOF frames, `FOLDS_CV`), `features`, `backbone.asof_matrix`, `baselines.LGB_PARAMS`, `hmm.transition_table`.
- Produces (module `gmst.analysis`):
  - `bin_edges(panel) -> dict[str, np.ndarray]` — `"prod"`: quartiles (25/50/75 %) of the positive finite `X["생산량"]` values on days ≤ 2021-08-31; `"temp"`: quartiles of finite `X["기온"]` on days ≤ 2021-08-31 (design statistics ≤ 08-31, DC10).
  - `prod_bin(x, edges) -> list[str]` — NaN → `"nan"`, 0 → `"0"`, else `f"Q{np.searchsorted(edges, x, side='right') + 1}"`; `quart_bin(x, edges) -> list[str]` — NaN → `"nan"`, else `Q1…Q4` the same way.
  - `error_by_condition(slots, panel, models, edges=None) -> pl.DataFrame` — columns `model, condition, bin, n, mae, rmse` over rows with finite `y_true` (variant `main`) for each model in `models`. Conditions: `prod_bin` (slot production), `op` (`op`/`nonop`), `daytype`, `hour` (0–23 as strings), `temp_bin`, `boundary` (`after_off` if `op[d−1] == 0`, else `before_off` if `op[d+1] == 0`, else `other`), and when the model has states: `state_filt`, `state_smooth` (bins `"1".."K"`), `transition` (`switch` if the slot's smoothed state differs from the next slot's within the day, else `stable`). MAE with `y_median`, RMSE with `y_mean` (FR-90, v2 §13.1–§13.3).
  - `classify(alarm, event) -> list[str]` — `TP/FP/FN/TN`.
  - `fn_fp(days, panel, p_star) -> tuple[pl.DataFrame, pl.DataFrame]` — rows = `usable_peak` rows of CV folds, variant `main`, per model present, per threshold `C50/C75/C90`, per `p_star_kind ∈ {"p05", "pstar"}`: alarm = `risk_platt ≥ 0.5` or `≥ p_star[(model, "main", fold)][C]`. Table columns `model, C, p_star_kind, date, kind, daytype, op_prev, op_next, prev_day_max, temp_mean, prod_sum, risk_cal, M_true` (`prev_day_max` = nanmax `Y[d−1]`, `temp_mean` = mean `기온[d]`, `prod_sum` = `sum(생산량[d]) / 4`). Summary columns `model, C, p_star_kind, group, stat, value, n` with `group ∈ {FN, FP, all}` and stats `share_wk, share_sat, share_sun, share_op_prev0, share_op_next0, mean_prev_day_max, mean_temp, mean_prod_sum` (`n` = group size) (FR-91, v2 §13.4). The representative threshold is C90, reported on pooled f1–f4 (13 events).
  - `gap_scores(X, y, day, event, n_random=5, seed=0, rounds=100, max_events=None) -> dict` — LightGBM L2 (`{**LGB_PARAMS, "objective": "regression"}`): `random5` = day-level random 5-fold (fold of each unique day = its position in `np.random.default_rng(seed).permutation(unique days)` mod 5) MAE over all rows; `loeo` = leave-one-event-out MAE over the rows of the events used (sorted event ids, first `max_events` if given); returns `{"random5", "loeo", "n", "n_events"}`. Marker: `# ponytail: 누수 격차는 L2 한 모델로만 측정, 분위수·일모델 격차가 필요하면 확장 (FR-92)`.
  - `leakage_gap(tau, half_life, rounds=100, max_events=None) -> pl.DataFrame` — columns `scheme, mae, n, n_folds`; rows `random5` then `loeo`; data = `load_panel(include_copies=True)` (sealed) days ≤ 2021-08-31 with ≥ 1 finite Y; features = `lgbm_rows(..., "A+", asof_matrix(...))`; events = `days.event_id` (FR-92, v2 §15.3).
  - `transitions(state, panel, fold) -> pl.DataFrame` = `hmm.transition_table(state["model"], panel, role_idx(panel, fold, "train"), state["protocol"])` (FR-93).

- [ ] **Step 1: Write the failing test** — create `tests/test_analysis.py`:

```python
import numpy as np
import polars as pl
import pytest

from gmst import analysis as an
from gmst import baselines as bl
from gmst import evaluate as ev
from gmst import features as ft
from gmst import hmm


@pytest.fixture(scope="module")
def panel():
    return ft.load_panel()


def test_classify():
    assert an.classify(np.array([1, 1, 0, 0], bool), np.array([1, 0, 1, 0], bool)) == ["TP", "FP", "FN", "TN"]


def test_bins():
    assert an.prod_bin(np.array([0.0, 5.0, 10.0, 25.0, 99.0, np.nan]), np.array([10.0, 20.0, 30.0])) == [
        "0", "Q1", "Q2", "Q3", "Q4", "nan"]
    assert an.quart_bin(np.array([-5.0, 0.0, 15.0, 30.0]), np.array([0.0, 10.0, 20.0])) == ["Q1", "Q2", "Q3", "Q4"]


def test_bin_edges_use_design_range(panel):
    e = an.bin_edges(panel)
    assert e["prod"].shape == (3,) and e["temp"].shape == (3,) and (np.diff(e["prod"]) >= 0).all()
    p2 = {**panel, "X": {k: v.copy() for k, v in panel["X"].items()}}
    late = [i for i, d in enumerate(panel["dates"]) if d.month == 9]
    p2["X"]["기온"][late] = 99.0
    assert np.array_equal(an.bin_edges(p2)["temp"], e["temp"])


def test_error_by_condition_small(panel):
    slots = ev.rolling_origin(bl.b0p_model(), panel, folds=("f1",))[0]
    t = an.error_by_condition(slots, panel, ["B0p"])
    assert t.columns == ["model", "condition", "bin", "n", "mae", "rmse"]
    assert {"prod_bin", "op", "daytype", "hour", "temp_bin", "boundary"} <= set(t["condition"].unique())
    assert t.filter(pl.col("condition") == "hour")["n"].sum() == 1152
    assert set(t.filter(pl.col("condition") == "boundary")["bin"].unique()) <= {"after_off", "before_off", "other"}


def test_fn_fp_small(panel):
    days = ev.rolling_origin(bl.b0p_model(), panel, folds=("f1",))[1]
    days = days.with_columns([pl.col(f"risk_raw_C{c}").alias(f"risk_platt_C{c}") for c in (50, 75, 90)])
    tbl, summ = an.fn_fp(days, panel, {("B0p", "main", "f1"): {"C50": 0.5, "C75": 0.5, "C90": 0.5}})
    assert tbl.columns == ["model", "C", "p_star_kind", "date", "kind", "daytype", "op_prev", "op_next",
                           "prev_day_max", "temp_mean", "prod_sum", "risk_cal", "M_true"]
    assert tbl.height == 12 * 3 * 2 and set(tbl["kind"].unique()) <= {"TP", "FN", "FP", "TN"}
    assert summ.columns == ["model", "C", "p_star_kind", "group", "stat", "value", "n"]


def test_gap_scores_detects_duplicate_leakage():
    rng = np.random.default_rng(0)
    rows, ys, days, events = [], [], [], []
    for e in range(30):
        z, u = rng.normal(size=3), rng.normal(0, 3.0)
        for c in range(3):
            for q in range(24):
                rows.append([q, *z])
                ys.append(np.sin(q / 4) + u)
                days.append(e * 3 + c)
                events.append(e)
    out = an.gap_scores(np.array(rows), np.array(ys), np.array(days), np.array(events), rounds=100)
    assert out["random5"] < 0.8 * out["loeo"] and out["n_events"] == 30


def test_transitions_wrapper(panel):
    tr = ft.role_idx(panel, "f4", "train")
    t = an.transitions({"model": hmm.CondHMM(K=3, dz=14, seed=0), "protocol": "A+"}, panel, "f4")
    cells = {(int(panel["op"][i]), int(panel["dtype"][i])) for i in tr}
    assert t.columns == ["op", "daytype", "hour", "from_state", "p_to_high"] and t.height == len(cells) * 24 * 3
```

- [ ] **Step 2: Run to verify it fails** — `uv run pytest tests/test_analysis.py -q` → `ModuleNotFoundError: No module named 'gmst.analysis'`.
- [ ] **Step 3: Implement** `gmst/analysis.py`.
- [ ] **Step 4: Run to verify it passes** — `uv run pytest tests/test_analysis.py -q` → `7 passed`.
- [ ] **Step 5: Full suite and commit**
```bash
git add gmst/analysis.py tests/test_analysis.py
git commit -m "feat: error-by-condition, FN/FP condition tables, augmentation leakage gap (random 5-fold vs LOEO), transition table"
```

---

### Task 18: One-command runner (stages ①–⑭), improvement-loop driver, smoke mode, sealed final stage, submission files

**Files:**
- Create: `gmst/run_all.py`, `tests/test_run_all.py`

**Interfaces:**
- Consumes: every module above.
- Produces (module `gmst.run_all`):
  - `FULL = {"folds": ("f1", "f2", "f3", "f4"), "max_epochs": 300, "patience": 20, "N": 2000, "B": 2000, "rounds": 100, "loeo_max_events": None, "scen_max_days": None}`; `SMOKE = {"folds": ("f4",), "max_epochs": 2, "patience": 20, "N": 50, "B": 20, "rounds": 10, "loeo_max_events": 5, "scen_max_days": 2}` (FR-99).
  - `PRED_COLS = ["datetime", "track", "model", "y_mean", "y_median", *QCOLS, "M_hat_median", "M_hat_mean", "peak_time_mode", "risk_C50", "risk_C75", "risk_C90", "C50", "C75", "C90"]` (33, FR-96).
  - `LADDER = ("H", "IO", "KAN")` — the pre-registered improvement-loop order (US-021, FR-110); `LOOP = ()` — the variants implemented so far (Tasks 21, 22, 23 set it to `("H",)`, `("H", "IO")`, `("H", "IO", "KAN")`); `MODEL_ID = {"main": "B4", "H": "B4-H", "IO": "B4-IO", "KAN": "B4-KAN"}` (FR-96).
  - `main(argv=None) -> int` with flags `--smoke`, `--no-final` (stop after stage ⑫), `--occ-floor` (turn on the HMM occupancy floor, FR-68), `--package` (Task 19), `--out DIR` (default `RESULTS`). `if __name__ == "__main__": raise SystemExit(main())`.
  - Every stage prints exactly one line `[NN] <name> <seconds:.1f>s device=<DEVICE.type>` (FR-94). Stage names in order: `preprocess, splits, oof, calibration, metrics, risk_check, bootstrap, loop, selection, state_stability, analysis, scenarios, final, submission`.

**Stages (FR-94 as amended, FR-95, FR-110):**
1. `preprocess` — `preprocess.run(out)` (writes `data_audit.json`).
2. `splits` — `splits.run()`.
3. `oof` — `panel`, `panel_copies`, `panel_suspect` via `load_panel`; `tau, h, table = backbone.select_tau(panel, folds)` → `backbone_tau.csv`; `lag7_skip_table(panel)` → `lag7_skip.csv`; then `rolling_origin` over `folds` for: main `B0, B0p, BB, B1, B2, B3, B4`; `copies` (B1, B4 on `panel_copies`); `suspect` (B1, B4 on `panel_suspect`); `protoA` (B1 and B4 with `protocol="A"`); `K2`, `K4` (B4); `re` (B4 `re=True`, `states=` the B4 main states — no retraining; diagnostic only, FR-75); `AWstar` (B1 `protocol="A+W*"`); `B` (B4 `protocol="B"`, `train_from=date(2021, 7, 1)`, folds = `folds ∩ (f2, f3, f4)`). HMM factories get `tau, half_life, max_epochs, patience, N, occ_floor`; B1 gets `rounds`. Keep the returned states of B4 main and SCENARIO.
4. `calibration` — `calibrate_oof(days, "platt", inner)` then `"iso"`; `p_star = p_star_table(inner)`; write `oof_slots.csv` (SLOT_COLS), `oof_days.csv` (DAY_COLS), `inner_days.csv` (DAY_COLS) (rewritten after stage 8 so they include the loop variants).
5. `metrics` — `point_metrics` + `risk_metrics(add_climatology(days, panel), panel, p_star)` + HMM `nll` rows (per fold: `value = state["val_nll"]`, `n` = observed validation points; `pooled` = n-weighted mean) → `metrics.csv` (rewritten after stage 8).
6. `risk_check` — `risk_check` on models B1–B4 (all their variants) → `risk_check.json` (NaN → null).
7. `bootstrap` — `compare(B4 main, B1 main, "B4-B1", ("mae", "crps", "brier_mean"), B)` and `compare(B1 AWstar, B1 main, "AWstar-main", ("mae", "brier_mean"), B)` → `bootstrap.csv` (`comparison, metric, delta, ci_lo, ci_hi, dm_stat, dm_p, n_days`).
8. `loop` — the pre-registered improvement loop (US-021, FR-107–FR-110; v2 §9.6, §15.6, A33–A36). `candidates = {"main": {...}}` for B4 main (`mae` = pooled f1–f4 MAE, `brier_mean` = pooled mean Platt Brier over C50/C75/C90, `gate_met = gate_pass(boot, "B4-B1")`, `gap` = the `B4-B1` mae/brier_mean `[delta, lo, hi]`); `adopted = []` (components of the current best). For `v` in `LADDER`: if any candidate so far has `gate_met` → stop (status of the rest `skipped_gate_met`); elif `v not in LOOP` → status `not_implemented` (the 2026-10-03 stop rule is enforced by not implementing later tasks); else tune its inner-window hyper-parameter on top of `adopted` (IO: `select_io_lambda` → `io_lambda.csv`; KAN: `select_kan` → `kan_grid.csv`), run `rolling_origin` for `hmm_model("B4", …, **flags(adopted + [v]))` as variant `v` (`flags` maps `H` → `emission_source="b1"`, `IO` → `decoder="io", io_lambda=<chosen>`, `KAN` → `kan=True, kan_M=<chosen>, kan_lambda=<chosen>`), calibrate, compare vs B1 as `"<MODEL_ID[v]>-B1"`, add `candidates[v]`; if `select_variant({best, v})` picks `v` (lower MAE, tie Brier) → `adopted.append(v)` ("채택", A34). Append the loop rows to the OOF frames, rewrite `oof_*.csv`, `inner_days.csv`, `metrics.csv`, `risk_check.json`, `bootstrap.csv`.
9. `selection` — `sel = select_variant(candidates)` (US-021); `final_epochs = hmm.final_epochs([best_epoch of the submitted variant per fold])`. `selection.json` = `{"submitted": MODEL_ID[sel["submitted"]], "submitted_variant", "gate_met", "tried", "criteria": {"mae": [delta, lo, hi], "brier_mean": [...]}, "reported": {"crps": [...]}, "remaining_gap_to_b1", "candidates": [{"variant", "model_id", "components", "status", "mae", "brier_mean", "gate_met"} for main + LADDER in order], "n_candidates_run", "tau", "half_life", "io_lambda" (float or null), "kan": {"M", "lambda_spl"} or null, "final_epochs", "p_star": {model: {variant: {fold: {C50, C75, C90}}}}, "rule": "B1 = benchmark; submit the select_variant winner among gate-passing B4 variants, else the lowest pooled OOF MAE (tie: mean Brier)", "final_protocol": "A+", "final_features": features.SLOT_FEATURES + features.DAY_FEATURES}` (FR-57 as amended, FR-110). `criteria` and `remaining_gap_to_b1` are the submitted variant's `-B1` CIs. B1 is never submitted.
10. `state_stability` — `hmm.state_stability(panel, "f1", (0, 1, 2), tau=tau, half_life=h, max_epochs, patience, occ_floor, device=DEVICE)` → `state_stability.csv` (`seed_a, seed_b, occ_maxdiff, delta_maxdiff, same_order, same_conclusion, agreement`); if `collapsed`, print `occupancy collapse (<0.02): rerun with --occ-floor`.
11. `analysis` — `error_by_condition` (the submitted variant, B4 main and the B1 benchmark) → `error_by_condition.csv`; `fn_fp` (same models) → `fn_fp_days.csv`, `fn_fp_summary.csv`; `leakage_gap(tau, h, rounds, loeo_max_events)` → `leakage_gap.csv`; `transitions(B4 main state of the last fold, panel, that fold)` → `hmm_transitions.csv`.
12. `scenarios` — `run_scenarios(panel, SCENARIO states, N, max_days=scen_max_days)` → `scenarios.csv`, `scenarios_month.csv`. With `--no-final`, return 0 here.
13. `final` — the only unseal in the package: `panel_open = features.load_panel(unseal=True)`. Models `B0`, `B0p`, `B1` (A+, benchmark only) and the **submitted B4 variant** (A+, its adopted flags, `final_epochs`, `io_lambda`/`kan` if used) — the sealed window is used for this one B4 variant only (FR-94 ⑬); for each: `slots_t, days_t, states_t, inner_t = rolling_origin(model, panel_open, folds=("test",))`; issued predictions `pred[d] = model["predict"](states_t["test"], panel_open, d)` for the 14 test days (96-slot issued values, FR-96); calibrate `days_t` and `inner_t` with `calibrate_oof(..., ref=<main OOF days of the same model/variant>)`; `p_star_test = p_star_table(inner_t)`.
14. `submission` — `test_predictions.csv` for the submitted variant: 1,344 rows, `PRED_COLS`, `datetime` `%Y.%m.%d %H:%M:%S`, `track = "MAIN"`, `model` = its model ID (`B4`, `B4-H`, `B4-IO` or `B4-KAN`), slot columns from the issued prediction, daily columns repeated over the day's 96 rows (`M_hat_*` issued, `peak_time_mode` as `HH:MM`, `risk_C*` = Platt-calibrated issued `risk_raw` with the ref calibrator then `np.minimum.accumulate`, `C*` = test thresholds (186.5, 198, 210.7)). No `test_predictions_B4_module.csv` (FR-98 deleted). `eval_mask.csv` = `datetime, is_missing` from `panel_open["is_missing"]` (FR-97). `test_metrics.csv` = `point_metrics` + `risk_metrics` of the four final models on fold `test` (B1 kept for comparison). Unless `--smoke`, refresh `ROOT/requirements.txt` with `write_requirements()` (Task 19; skip silently if it returns False).

- [ ] **Step 1: Write the failing test** — create `tests/test_run_all.py`:

```python
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from gmst import evaluate as ev

ROOT = Path(__file__).resolve().parents[1]
PRED_COLS = ["datetime", "track", "model", "y_mean", "y_median", *ev.QCOLS, "M_hat_median", "M_hat_mean",
             "peak_time_mode", "risk_C50", "risk_C75", "risk_C90", "C50", "C75", "C90"]
FILES = ["data_audit.json", "backbone_tau.csv", "lag7_skip.csv", "oof_slots.csv", "oof_days.csv", "inner_days.csv",
         "metrics.csv", "risk_check.json", "bootstrap.csv", "selection.json", "state_stability.csv",
         "error_by_condition.csv", "fn_fp_days.csv", "fn_fp_summary.csv", "leakage_gap.csv", "hmm_transitions.csv",
         "scenarios.csv", "scenarios_month.csv", "test_predictions.csv", "eval_mask.csv", "test_metrics.csv"]
STAGES = ["preprocess", "splits", "oof", "calibration", "metrics", "risk_check", "bootstrap", "loop", "selection",
          "state_stability", "analysis", "scenarios", "final", "submission"]


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    out = tmp_path_factory.mktemp("results")
    r = subprocess.run([sys.executable, "-m", "gmst.run_all", "--smoke", "--out", str(out)], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    return out, r.stdout


def read(out, name):
    return pl.read_csv(out / name, infer_schema_length=None)


def test_files_exist(smoke):
    out, _ = smoke
    assert not [f for f in FILES if not (out / f).exists()]


def test_stage_log_and_seal_order(smoke):
    _, log = smoke
    lines = [l for l in log.splitlines() if l.startswith("[")]
    assert [l.split()[1] for l in lines] == STAGES
    assert all(l.split()[-1].startswith("device=") for l in lines)


def test_oof_and_metric_schemas(smoke):
    out, _ = smoke
    assert read(out, "oof_slots.csv").columns == ev.SLOT_COLS
    assert read(out, "oof_days.csv").columns == ev.DAY_COLS
    for name in ("metrics.csv", "test_metrics.csv"):
        assert read(out, name).columns == ["model", "variant", "fold", "stratum", "metric", "value", "n"]
    assert set(read(out, "test_metrics.csv")["fold"].unique()) == {"test"}
    assert read(out, "bootstrap.csv").columns == ["comparison", "metric", "delta", "ci_lo", "ci_hi", "dm_stat",
                                                  "dm_p", "n_days"]
    variants = set(read(out, "oof_days.csv")["variant"].unique())
    assert variants >= {"main", "copies", "suspect", "protoA", "K2", "K4", "re", "AWstar", "B"}


def test_prediction_file(smoke):
    out, _ = smoke
    t = read(out, "test_predictions.csv")
    sel = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    assert t.columns == PRED_COLS and t.height == 1344
    assert t["datetime"][0] == "2021.09.01 00:00:00" and t["datetime"][-1] == "2021.09.14 23:45:00"
    assert set(t["track"].unique()) == {"MAIN"} and set(t["model"].unique()) == {sel["submitted"]}
    assert np.allclose(t.select("C50", "C75", "C90").to_numpy(), [186.5, 198.0, 210.7])
    assert (np.diff(t.select(ev.QCOLS).to_numpy(), axis=1) >= 0).all()
    r = t.select("risk_C50", "risk_C75", "risk_C90").to_numpy()
    assert ((r >= 0) & (r <= 1)).all() and (r[:, 0] >= r[:, 1]).all() and (r[:, 1] >= r[:, 2]).all()
    assert "is_missing" not in t.columns and t["peak_time_mode"].str.contains(r"^\d{2}:\d{2}$").all()
    assert sel["final_protocol"] == "A+" and not [f for f in sel["final_features"] if f.startswith("ob_")]


def test_eval_mask(smoke):
    out, _ = smoke
    m = read(out, "eval_mask.csv")
    assert m.columns == ["datetime", "is_missing"] and m.height == 1344
    assert m.filter(pl.col("is_missing"))["datetime"].to_list() == ["2021.09.08 12:00:00", "2021.09.08 12:15:00"]


def test_selection_is_b4_family(smoke):
    out, _ = smoke
    sel = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    assert sel["submitted"] in ("B4", "B4-H", "B4-IO", "B4-KAN") and sel["submitted_variant"] in ("main", "H", "IO", "KAN")
    assert {"submitted", "gate_met", "tried", "criteria", "reported", "remaining_gap_to_b1", "candidates",
            "n_candidates_run", "tau", "half_life", "io_lambda", "kan", "final_epochs", "p_star", "rule",
            "final_protocol", "final_features"} <= set(sel)
    assert set(sel["criteria"]) == {"mae", "brier_mean"} and set(sel["reported"]) == {"crps"}
    assert [c["variant"] for c in sel["candidates"]] == ["main", "H", "IO", "KAN"]
    assert {c["status"] for c in sel["candidates"]} <= {"run", "skipped_gate_met", "not_implemented"}
    assert sel["tried"] == [c["variant"] for c in sel["candidates"] if c["status"] == "run"]
    assert sel["n_candidates_run"] == len(sel["tried"]) and sel["tried"][0] == "main"
    assert sel["submitted_variant"] in sel["tried"]
    assert not (out / "test_predictions_B4_module.csv").exists()
    boot = read(out, "bootstrap.csv")
    assert {"B4-B1", "AWstar-main", f"{sel['submitted']}-B1"} <= set(boot["comparison"].unique())
```

- [ ] **Step 2: Run to verify it fails** — `uv run pytest tests/test_run_all.py -q` → the fixture fails with `No module named gmst.run_all`.
- [ ] **Step 3: Implement** `gmst/run_all.py`. Keep the stage bodies as plain sequential code in `main` (one function per stage only where a stage is reused); no config files.
- [ ] **Step 4: Run to verify it passes** — `uv run pytest tests/test_run_all.py -q` → `6 passed` (one smoke run, a few minutes). Also `uv run pytest tests/test_style.py -q` (the `unseal=True` literal must be in `run_all.py` only).
- [ ] **Step 5: Full suite and commit**
```bash
git add gmst/run_all.py tests/test_run_all.py
git commit -m "feat: one-command runner (14 logged stages incl. improvement-loop driver), smoke mode, sealed final stage, submission files"
```

---

### Task 19: requirements.txt, source package zip, ignore rules

**Files:**
- Modify: `gmst/run_all.py` (add `write_requirements`, `build_package`, `--package`), `.gitignore`, `tests/test_run_all.py` (append)
- Create: `requirements.txt` (generated)

**Interfaces:**
- `run_all.write_requirements(path=ROOT / "requirements.txt") -> bool` — if `shutil.which("uv")` is None return False (marker `# ponytail: uv가 없으면 커밋된 requirements.txt 유지, 평가 환경에서 재생성이 필요하면 importlib.metadata로 직접 핀 (FR-100)`); else run `uv export --no-hashes --no-dev --no-emit-project --format requirements.txt` in `ROOT` and write `--extra-index-url https://download.pytorch.org/whl/cu126\n` + its stdout; return True (FR-100).
- `run_all.build_package(dest=ROOT / "dist" / "kamp_power_src.zip") -> Path` — `zipfile` (deflated) with arcnames relative to `ROOT` (forward slashes): `gmst/**/*.py`, `tests/**/*.py`, `notebooks/*.ipynb`, `results/**` (if present), the CSVs directly in `DATA` (raw + two script outputs), `requirements.txt`, `README.md`, `pyproject.toml`, `uv.lock`, `PONYTAIL-DEBT.md` (if present; v2 §20 ②). Never `.env`, `.venv/`, `document/`, `tasks/`, `claudedocs/`, `.sdd/`, `__pycache__/`, `dist/` (FR-102). `main(["--package"])` builds the zip and returns 0 without running stages.
- `.gitignore` gains `dist/`, `*.log`, `results_repro/`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_run_all.py`:

```python
import zipfile


def test_requirements():
    lines = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "--extra-index-url https://download.pytorch.org/whl/cu126"
    pins = [l.split(";")[0].strip() for l in lines if "==" in l and not l.lstrip().startswith("#")]
    names = {p.split("==")[0].lower() for p in pins}
    assert "torch==2.14.0+cu126" in pins
    assert {"lightgbm", "polars", "numpy", "matplotlib"} <= names
    assert not names & {"pytest", "requests"}


def test_package():
    r = subprocess.run([sys.executable, "-m", "gmst.run_all", "--package"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    names = zipfile.ZipFile(ROOT / "dist" / "kamp_power_src.zip").namelist()
    for need in ("README.md", "requirements.txt", "pyproject.toml", "uv.lock", "gmst/run_all.py",
                 "tests/test_run_all.py", "5. 자원 최적화 AI 데이터셋/okm_augumented_2021.csv",
                 "5. 자원 최적화 AI 데이터셋/okm_15min_2021.csv", "5. 자원 최적화 AI 데이터셋/okm_cv_splits_2021.csv"):
        assert need in names, need
    if (ROOT / "results").exists():
        assert any(n.startswith("results/") for n in names)
    bad = [n for n in names if n == ".env" or "__pycache__" in n
           or n.startswith((".venv/", "document/", "tasks/", "claudedocs/", ".sdd/", "dist/"))]
    assert not bad, bad
```

- [ ] **Step 2: Run to verify they fail** — `uv run pytest tests/test_run_all.py -q -k "requirements or package"` → `FileNotFoundError` for `requirements.txt` and a non-zero `--package` exit.
- [ ] **Step 3: Implement** `write_requirements`, `build_package`, `--package`; update `.gitignore`; generate the file once: `uv run python -c "from gmst.run_all import write_requirements; print(write_requirements())"` → `True`.
- [ ] **Step 4: Run to verify they pass** — `uv run pytest tests/test_run_all.py -q` → `8 passed`.
- [ ] **Step 5: Full suite and commit**
```bash
git add gmst/run_all.py tests/test_run_all.py .gitignore requirements.txt
git commit -m "feat: requirements.txt from uv export with cu126 index line, source package zip"
```

---

### Task 20: Development run (no unseal) and the B1-benchmark gate check

**Files:** none created; produces `results/` locally (not committed yet) and the gate decision that starts or skips the improvement loop.

**Rules:** no unseal (`--no-final`). The gate (`gate_pass`, US-008) is the success criterion of the B4 family against the B1 benchmark (user decision, A33): ΔMAE and Δmean-Brier (variant − B1, Platt, C50/C75/C90) day-block-bootstrap 95 % CI upper bounds both < 0 on the 54 OOF days. It is read from `results/selection.json` — never from test-window data.

- [ ] **Step 1: Full suite** — `uv run pytest -q` → all passed.
- [ ] **Step 2: Development run** — `rm -rf results && uv run python -m gmst.run_all --no-final 2>&1 | tee run_dev.log` (≈ 1 h on the A6000; run it in the background and poll). Expected: exit 0, stage lines `[01]`…`[12]`, `results/selection.json` whose `candidates` list shows `main` as `run` and `H`, `IO`, `KAN` as `not_implemented` (or `skipped_gate_met`).
- [ ] **Step 3: Read the gate** — `uv run python -c "import json; s=json.load(open('results/selection.json')); print(s['gate_met'], s['candidates'], s['remaining_gap_to_b1'])"`.
  - If `gate_met` is true for `main`: the improvement loop is **not triggered**; Tasks 21–23 are skipped (their candidates stay `skipped_gate_met`, which the report discloses). Go to Task 24.
  - Otherwise start Task 21 (loop ①). Stop rule for the whole loop (v2 §15.6, A33): stop as soon as a dev run shows `gate_met` for a variant, or when the date passes **2026-10-03**, whichever comes first; then go to Task 24.
- [ ] **Step 4: Other OOF checks (reported, not gates)**
  - B1 vs B0′ (success metric): `uv run python -c "import polars as pl; m=pl.read_csv('results/metrics.csv'); print(m.filter((pl.col('fold')=='pooled')&(pl.col('stratum')=='all')&(pl.col('variant')=='main')&(pl.col('metric')=='mae')&pl.col('model').is_in(['B0p','B1'])))"`. If B1 ≥ B0′, the LightGBM ponytail trigger fired: record it for the ledger (Task 26) and report it to the controller. Hyper-parameter search is a PRD non-goal, so do not tune without a user decision.
  - State collapse: if `run_dev.log` contains `occupancy collapse`, use `--occ-floor` in every later run (FR-68, A21).
  - `risk_check.json` (B3/B4 discrimination, the `re` diagnostic) and `state_stability.csv` are reported as they are.
- [ ] **Step 5: Report** the gate result, `remaining_gap_to_b1`, the B1-vs-B0′ line and the collapse flag to the controller (no commit; `results/` is committed in Task 24).

---

### Task 21: Improvement loop ① — B4-H hybrid centre (variant `H`, model ID `B4-H`)

**Do this task only if Task 20 found the gate not met and the date is ≤ 2026-10-03.** (US-021, FR-107, v2 §9.6 ①, A34.)

**Files:**
- Modify: `gmst/baselines.py` (append `b1_centre`), `gmst/hmm.py` (`hmm_model(emission_source=…)`), `gmst/run_all.py` (`LOOP = ("H",)`), `tests/test_hmm.py` (append), `tests/test_leakage.py` (append), `tests/test_run_all.py` (append)

**Interfaces:**
- `baselines.b1_centre(panel, train_idx, tau, half_life, C, protocol="A+", rounds=100, n_blocks=5) -> np.ndarray` (257, 96) — the emission centre `m_t = Ŷ^{B1}_t` = B1's τ = 0.5 slot forecast (`predict_b1(...)["y_median"]`, A32 features unchanged). Every day **not** in `train_idx` gets `predict_b1(fit_b1(panel, train_idx, protocol, (tau, half_life), C, rounds), panel, d)["y_median"]` (the fold-safe forecast the fold's own B1 issues at 00:00). The days of `train_idx` get **cross-fitted** values: `np.array_split(np.sort(train_idx), n_blocks)` contiguous chunks, each chunk's days predicted by `fit_b1` trained on the other chunks. `n_blocks=1` means "no cross-fitting" (one fit on all of `train_idx` predicts every day; used only as the in-sample comparison in the test). Marker: `# ponytail: 학습일 중심은 5개 연속 블록 교차적합, 시간순 확장창이 필요하면 일별 as-of 재적합 (v2 §9.6 ①)` (**[plan pin]**, see Spec issues).
- `hmm.hmm_model(..., emission_source="backbone", rounds=100)` — `emission_source="b1"` → `m_full = b1_centre(panel, tr, tau, half_life, C, protocol, rounds)` and `m_in = b1_centre(panel, setdiff(tr, inn), …)` replace `fold_backbone`; δ, σ, φ, transitions, masks, the R10 procedure and MC are unchanged (`μ^H_{t,k} = Ŷ^{B1}_t + δ_{k,op}`, FR-107). (**[plan pin]**: the PRD names `emission_source` on the `CondHMM` constructor; `m_t` is data built outside the torch module, so the switch lives in the `hmm_model` factory that builds it.)
- `run_all.LOOP = ("H",)`; `run_all` builds the variant with `emission_source="b1"`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_hmm.py`:

```python
from gmst import baselines as bl


def test_b1_centre_equals_predict_b1_on_validation_days():
    p = ft.load_panel()
    tr = ft.role_idx(p, "f4", "train")
    C, _ = ev.thresholds(p, tr)
    m = bl.b1_centre(p, tr, 10.0, 60, C, rounds=20)
    b1 = bl.fit_b1(p, tr, "A+", (10.0, 60), C, rounds=20)
    for d in ft.role_idx(p, "f4", "val")[:3]:
        assert np.array_equal(m[d], bl.predict_b1(b1, p, int(d))["y_median"])


def test_b1_centre_is_cross_fitted_on_train_days():
    p = ft.load_panel()
    tr = ft.role_idx(p, "f4", "train")
    C, _ = ev.thresholds(p, tr)
    m = bl.b1_centre(p, tr, 10.0, 60, C, rounds=50)
    full = bl.b1_centre(p, tr, 10.0, 60, C, rounds=50, n_blocks=1)
    y = p["Y"][tr]
    f = np.isfinite(y)
    assert np.isfinite(m[tr]).all()
    assert np.mean(np.abs(m[tr][f] - y[f])) > np.mean(np.abs(full[tr][f] - y[f]))


def test_hybrid_model_uses_b1_centre():
    p = ft.load_panel()
    tr = ft.role_idx(p, "f4", "train")
    C, _ = ev.thresholds(p, tr)
    model = hmm.hmm_model("B4", emission_source="b1", tau=10.0, half_life=60, rounds=20, max_epochs=2, N=50,
                          device="cpu")
    st = model["fit"](p, "f4", C)
    assert np.array_equal(st["m"], bl.b1_centre(p, tr, 10.0, 60, C, rounds=20))
    out = model["predict"](st, p, p["dates"].index(date(2021, 8, 18)))
    assert out["paths"].shape == (50, 96) and np.isfinite(out["y_median"]).all()
```

Append to `tests/test_leakage.py`:

```python
def test_b4h_ignores_post_origin_values_and_later_values(panel):
    C, _ = ev.thresholds(panel, ft.role_idx(panel, "f4", "train"))
    m = hmm.hmm_model("B4", emission_source="b1", rounds=20, **HMM_KW)
    st = m["fit"](panel, "f4", C)
    d = panel["dates"].index(D0)
    same(m["predict"](st, panel, d), m["predict"](st, perturb(panel, d, d, d), d))
    tr = ft.role_idx(panel, "f4", "train")
    after = int(tr.max()) + 1
    st2 = m["fit"](perturb(panel, after, after, after), "f4", C)
    assert np.array_equal(st["m"][tr], st2["m"][tr])
    assert all(torch.equal(x, y) for x, y in zip(st["model"].state_dict().values(), st2["model"].state_dict().values()))
```

and to `tests/test_run_all.py`:

```python
def test_loop_registry_has_h():
    from gmst import run_all
    assert run_all.LADDER == ("H", "IO", "KAN") and run_all.LOOP[:1] == ("H",)
    assert run_all.MODEL_ID == {"main": "B4", "H": "B4-H", "IO": "B4-IO", "KAN": "B4-KAN"}
```

- [ ] **Step 2: Run to verify they fail** — `uv run pytest tests/test_hmm.py tests/test_leakage.py tests/test_run_all.py -q` → exactly the 5 new tests fail (`… has no attribute 'b1_centre'`, `unexpected keyword argument 'emission_source'`, `assert () == ('H',)`); all earlier tests pass.
- [ ] **Step 3: Implement** `b1_centre`, the `emission_source`/`rounds` arguments of `hmm_model`, and `LOOP = ("H",)`.
- [ ] **Step 4: Run to verify they pass** — the same command → all passed; then `uv run pytest -q` → all passed.
- [ ] **Step 5: Commit**
```bash
git add gmst/baselines.py gmst/hmm.py gmst/run_all.py tests/test_hmm.py tests/test_leakage.py tests/test_run_all.py
git commit -m "feat: improvement loop 1 - B4-H (B1 fold-safe median as emission centre, cross-fitted on train days)"
```
- [ ] **Step 6: Dev run and gate** — `rm -rf results && uv run python -m gmst.run_all --no-final 2>&1 | tee run_dev.log`; read `results/selection.json` (`gate_met`, `candidates`). If a variant met the gate or the date is past 2026-10-03 → go to Task 24; else Task 22.

---

### Task 22: Improvement loop ② — B4-IO conditional decoder (variant `IO`, model ID `B4-IO`)

**Do this task only if the gate is still not met after Task 21 and the date is ≤ 2026-10-03.** (US-021, FR-108, v2 §9.6 ②, A35; ledger "Ruling (B4-IO spec)".) B4-IO replaces the old day-random-effect and state × daytype-variance candidates; the `re` ablation stays a diagnostic only (FR-75).

**Files:**
- Modify: `gmst/hmm.py`, `gmst/run_all.py` (`LOOP = ("H", "IO")`, `io_lambda.csv`), `tests/test_hmm.py` (append), `tests/test_leakage.py` (append), `tests/test_run_all.py` (append)

**Interfaces and math (input–output HMM emission; transitions unchanged):**
- `hmm.U_COLS = ["one", "op", "sin1", "cos1", "sin2", "cos2", "op_sin1", "op_cos1", "sat", "sun", "hol", "ybar_prev", "ymax_prev"]` (13, US-021 order).
- `hmm.u_features(panel, protocol, m, day_idx) -> np.ndarray` (n, 96, 13) — `u_t = [1, op_d, sin(2πq/96), cos(2πq/96), sin(4πq/96), cos(4πq/96), op_d·sin(2πq/96), op_d·cos(2πq/96), 1[Sat], 1[Sun], hol_d, ybar_{d−1}, ymax_{d−1}]` with `op, hol` from `cal_flags(panel, protocol)`; `ybar_{d−1}` / `ymax_{d−1}` = mean / max of `Y[d−1] − m[d−1]` over day d−1's observed slots; both 0.0 when day d−1 has no observed slot or d = 0 (13 columns always). `hmm.prev_missing_days(panel, day_idx) -> int` counts those zero-filled days; `run_all` prints `ybar_prev_missing_days=<n>` in the `loop` stage (the PRD's audit count).
- `hmm.v_features(panel, day_idx) -> np.ndarray` (n, 96, 3) = `[1, sin(2πq/96), cos(2πq/96)]`.
- `CondHMM(..., decoder="const")` — `decoder="io"` replaces `rho, s, psi` by `a` (K, 13), `b` (K, 13), `c` (K, 3): emission parameters `a + b + c` = 87 for K = 3 (transitions unchanged, 90 for A+). Init: `a[k, 0]` / `a[k, 1]` = the op-0 value / (op-1 − op-0) difference of the base `rho` init; `b[k, 0]`, `b[k, 1]` likewise from `s`; `c[k, 0]` = the base `psi` init; all other entries 0; plus the seed-driven N(0, 0.01²) noise.
- `CondHMM.emission_t(m, opt, u=None, v=None) -> tuple[mu, sig, phi]` (each (…, K)). `decoder="const"`: `mu = m + δ[k, opt]`, `sig = σ[k, opt]`, `phi = phi()` broadcast. `decoder="io"`: `δ_{t,1} = a_1ᵀu_t`, `δ_{t,k} = δ_{t,k−1} + softplus(a_kᵀu_t)`, `mu = m + δ_t`, `sig = 1 + softplus(b_kᵀu_t)`, `phi = sigmoid(c_kᵀv_t)` (FR-108). `forward_logp`, `backward`, `filter_last`, `forecast`, `posthoc_states` switch to `emission_t` with time-varying φ everywhere (pairwise emission `N(y_t; μ_{t,k} + φ_{t,k}(y_{t−1} − μ_{t−1,i}), σ²_{t,k})`, stationary marginal `σ²_{t,k}/(1 − φ²_{t,k})`, MC `e_h = φ_{h,S_h} e_{h−1} + σ_{h,S_h} ε_h`, masked-e0 draw with the parameters of the last slot of day d−1). `forward_logp(model, y, obs, m, opt, z, u=None, v=None)` and `backward(…, u=None, v=None)` gain optional trailing arguments; `make_blocks(..., io=False)` adds `u` (B, 192, 13) and `v` (B, 192, 3) when `io=True`. All Task 11–14 tests keep passing unchanged.
- `hmm.io_penalty(model) -> Tensor` = `Σ a[:, H]² + Σ b[:, H]²` with `H = [2, 3, 4, 5, 6, 7, 11, 12]` (harmonic and yesterday-summary columns; intercept, `op`, `sat`, `sun`, `hol` and all of `c` are not penalised, US-021). `fit_blocks(..., penalty=None)` adds `penalty(model)` to the loss when given (a callable already multiplied by its λ).
- `hmm.select_io_lambda(panel, flags, folds, grid=(0.0, 0.01, 0.1, 1.0), **kw) -> tuple[float, pl.DataFrame]` — for each λ and fold: the inner phase of `fit_with_inner` (train on `train \ inner`, early stopping on the inner window) with `flags` (the adopted variants' factory flags) + `decoder="io"`; record its best inner NLL; pool over folds weighted by observed inner slots; return the argmin λ (ties → first) and the table `lambda_io, nll_inner` (→ `results/io_lambda.csv`, US-021). The folds' validation days are never used (R10; **[plan pin]**: inner NLL as the criterion, see Spec issues).
- `hmm_model(..., decoder="const", io_lambda=0.0)`: `decoder="io"` → `CondHMM(decoder="io")`, IO blocks, penalty `io_lambda · io_penalty`. `run_all`: for the `IO` step, `select_io_lambda` on top of the adopted flags, then run; `LOOP = ("H", "IO")`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_hmm.py`:

```python
def test_io_param_count_and_ordering():
    model = hmm.CondHMM(K=3, dz=14, decoder="io", seed=0).double()
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 90 + 87
    assert model.a.numel() + model.b.numel() + model.c.numel() == 87
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for prm in model.parameters():
            prm.add_(torch.randn(prm.shape, generator=g, dtype=prm.dtype))
    u = torch.randn(200, 13, generator=g, dtype=torch.float64)
    u[:, 0] = 1.0
    v = torch.tensor(hmm.v_features(ft.load_panel(), np.array([0, 1, 2]))[..., :].reshape(-1, 3)[:200])
    mu, sig, phi = model.emission_t(torch.zeros(200, dtype=torch.float64), torch.ones(200, dtype=torch.long), u, v)
    assert (mu[:, 1:] > mu[:, :-1]).all() and (sig >= 1).all() and ((phi > 0) & (phi < 1)).all()


def test_io_nests_b4_likelihood():
    base = rand_model(K=3, dz=14, seed=4)
    io = hmm.CondHMM(K=3, dz=14, decoder="io", seed=4).double()
    with torch.no_grad():
        io.W.copy_(base.W)
        for t in (io.a, io.b, io.c):
            t.zero_()
        io.a[:, 0] = base.rho[:, 1]
        io.b[:, 0] = base.s[:, 1]
        io.c[:, 0] = base.psi
    blk = {k: (v.double() if v.is_floating_point() else v) for k, v in synthetic_blocks(n=6).items() if k != "days"}
    g = torch.Generator().manual_seed(9)
    u = torch.randn(6, 192, 13, generator=g, dtype=torch.float64)
    u[..., 0], u[..., 1] = 1.0, 1.0                              # op = 1 in synthetic_blocks
    v = torch.randn(6, 192, 3, generator=g, dtype=torch.float64)
    v[..., 0] = 1.0
    a = hmm.forward_logp(base, blk["y"], blk["obs"], blk["m"], blk["opt"], blk["z"])[0]
    b = hmm.forward_logp(io, blk["y"], blk["obs"], blk["m"], blk["opt"], blk["z"], u, v)[0]
    assert torch.allclose(a, b, atol=1e-6)


def test_u_features_use_only_previous_day():
    p = ft.load_panel()
    d = p["dates"].index(date(2021, 8, 18))
    m = np.full((257, 96), 100.0)
    u = hmm.u_features(p, "A+", m, np.array([d]))
    assert u.shape == (1, 96, 13) and hmm.U_COLS[-2:] == ["ybar_prev", "ymax_prev"]
    assert u[0, 0, 11] == pytest.approx(np.nanmean(p["Y"][d - 1] - 100.0))
    assert u[0, 0, 12] == pytest.approx(np.nanmax(p["Y"][d - 1] - 100.0))
    for k in (d, d - 2):
        p2 = {**p, "Y": p["Y"].copy()}
        p2["Y"][k] += 50.0
        assert np.array_equal(hmm.u_features(p2, "A+", m, np.array([d])), u)
    p3 = {**p, "Y": p["Y"].copy()}
    p3["Y"][d - 1] += 50.0
    assert not np.array_equal(hmm.u_features(p3, "A+", m, np.array([d])), u)
    masked = p["dates"].index(date(2021, 7, 14))                 # d-1 = 07-13 is fully masked
    assert (hmm.u_features(p, "A+", m, np.array([masked]))[0, :, 11:] == 0).all()
    assert hmm.prev_missing_days(p, np.array([masked, d])) == 1


def test_io_penalty_skips_intercept_op_flags_and_phi():
    model = hmm.CondHMM(K=3, dz=14, decoder="io", seed=0)
    with torch.no_grad():
        for t in (model.a, model.b, model.c):
            t.zero_()
        model.a[:, [0, 1, 8, 9, 10]] = 5.0
        model.c[:] = 5.0
    assert hmm.io_penalty(model).item() == 0.0
    with torch.no_grad():
        model.b[0, 11] = 2.0
    assert hmm.io_penalty(model).item() == pytest.approx(4.0)


def test_select_io_lambda_table():
    p = ft.load_panel()
    lam, table = hmm.select_io_lambda(p, {}, folds=("f4",), grid=(0.0, 1.0), tau=10.0, half_life=60, max_epochs=2,
                                      device="cpu")
    assert table.columns == ["lambda_io", "nll_inner"] and table.height == 2 and lam in (0.0, 1.0)
```

Append to `tests/test_leakage.py`:

```python
def test_b4io_ignores_post_origin_values(panel):
    C, _ = ev.thresholds(panel, ft.role_idx(panel, "f4", "train"))
    m = hmm.hmm_model("B4", decoder="io", io_lambda=0.01, **HMM_KW)
    st = m["fit"](panel, "f4", C)
    d = panel["dates"].index(D0)
    same(m["predict"](st, panel, d), m["predict"](st, perturb(panel, d, d, d), d))
```

and to `tests/test_run_all.py`:

```python
def test_loop_registry_has_io():
    from gmst import run_all
    assert run_all.LOOP[:2] == ("H", "IO")
```

- [ ] **Step 2: Run to verify they fail** — `uv run pytest tests/test_hmm.py tests/test_leakage.py tests/test_run_all.py -q` → exactly the 7 new tests fail (`unexpected keyword argument 'decoder'`, missing `u_features` …); all earlier tests pass.
- [ ] **Step 3: Implement** the IO decoder, the `emission_t` refactor, the penalty hook, `select_io_lambda`, `prev_missing_days`, `LOOP = ("H", "IO")` and `io_lambda.csv` in `run_all`.
- [ ] **Step 4: Run to verify they pass** — same command → all passed (Tasks 11–14 and 21 tests unchanged and green); `uv run pytest -q` → all passed.
- [ ] **Step 5: Commit**
```bash
git add gmst/hmm.py gmst/run_all.py tests/test_hmm.py tests/test_leakage.py tests/test_run_all.py
git commit -m "feat: improvement loop 2 - B4-IO conditional decoder (delta, sigma, phi as functions of u_t, v_t)"
```
- [ ] **Step 6: Dev run and gate** — as Task 21 Step 6; if not met and ≤ 2026-10-03 → Task 23, else Task 24.

---

### Task 23: Improvement loop ③ — B4-KAN periodic-spline transition (variant `KAN`, model ID `B4-KAN`)

**Do this task only if the gate is still not met after Task 22 and the date is ≤ 2026-10-03.** (US-021, FR-109, v2 §9.6 ③, A36.)

**Files:**
- Modify: `gmst/hmm.py`, `gmst/run_all.py` (`LOOP = ("H", "IO", "KAN")`, `kan_grid.csv`), `tests/test_hmm.py` (append), `tests/test_leakage.py` (append), `tests/test_run_all.py` (append)

**Interfaces and math:**
- `hmm.cyclic_bspline(M) -> np.ndarray` (96, M) — `B_m(q) = b3(((q·M/96) − m) mod M)` with the cardinal cubic B-spline `b3(x) = x³/6` (0 ≤ x < 1), `(−3x³ + 12x² − 12x + 4)/6` (1 ≤ x < 2), `(3x³ − 24x² + 60x − 44)/6` (2 ≤ x < 3), `(4 − x)³/6` (3 ≤ x < 4), 0 otherwise. Partition of unity, non-negative, period 96, uniform knots.
- `z_features(panel, protocol, day_idx, kan=False, M=12)` — `kan=True` (protocol `A+` only, assert) drops the 6 harmonics and the 4 op×harmonic interactions and returns `[(1 − op)·B_1(q) … (1 − op)·B_M(q), op·B_1(q) … op·B_M(q), sat, sun, hol, op]` (2M + 4), i.e. one spline `g(q, op) = Σ_m c_{op,m} B_m(q)` per transition pair and operating state, plus the 4 linear 0/1 flags (FR-109).
- `CondHMM(..., intercept=True)` — the KAN mode uses `intercept=False` because each op spline already spans a constant (partition of unity); transitions then have `K(K−1)(2M + 4)` parameters (168 for K = 3, M = 12 — the v2 §9.6 count; **[plan pin]**, see Spec issues).
- `hmm.spline_penalty(model, M) -> Tensor` = `Σ_{i,c} Σ_{block ∈ {0, M}} Σ_{m=0}^{M−1} (W[i, c, block + (m+1) mod M] − W[i, c, block + m])²` (cyclic first differences inside each op block; flags unpenalised).
- `hmm.select_kan(panel, flags, folds, M_grid=(8, 12, 16), lambda_grid=(0.0, 0.01, 0.1, 1.0), **kw) -> tuple[int, float, pl.DataFrame]` — the `select_io_lambda` procedure over the 3 × 4 grid with `flags` + `kan=True`; table columns `M, lambda_spl, nll_inner` (→ `results/kan_grid.csv`, US-021); ties → first in grid order. If the chosen M is 8 or 16, print `kan grid edge: M=<M>` (A36). (**[plan pin]** of the 4-value λ grid, see Spec issues.)
- `hmm_model(..., kan=False, kan_M=12, kan_lambda=0.0)`: `kan=True` → `CondHMM(dz=2·kan_M + 4, intercept=False)`, KAN z features, penalty `kan_lambda · spline_penalty` (added to any IO penalty). `run_all`: for the `KAN` step, `select_kan` on top of the adopted flags, then run; `LOOP = ("H", "IO", "KAN")`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_hmm.py`:

```python
@pytest.mark.parametrize("M", [8, 12, 16])
def test_cyclic_bspline(M):
    B = hmm.cyclic_bspline(M)
    assert B.shape == (96, M) and (B >= 0).all() and np.allclose(B.sum(1), 1.0)
    assert B.max() == pytest.approx(2 / 3)
    assert np.allclose(np.roll(B, 96 // M, axis=0)[:, :-1], B[:, 1:])       # B_m(q - one knot) = B_{m+1}(q)


def test_kan_param_count_and_features():
    model = hmm.CondHMM(K=3, dz=2 * 12 + 4, intercept=False, seed=0)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 168 + 15
    p = ft.load_panel()
    i = p["dates"].index(date(2021, 8, 18))
    z = hmm.z_features(p, "A+", np.array([i]), kan=True, M=12)
    assert z.shape == (1, 96, 28)
    op = int(p["op"][i])
    assert np.allclose(z[0, :, 12 * op:12 * op + 12], hmm.cyclic_bspline(12))
    assert np.allclose(z[0, :, 12 * (1 - op):12 * (1 - op) + 12], 0.0) and np.allclose(z[0, :, -1], op)


def test_spline_penalty():
    model = hmm.CondHMM(K=3, dz=2 * 8 + 4, intercept=False, seed=0)
    with torch.no_grad():
        model.W.zero_()
        model.W[..., :16] = 3.0
        model.W[..., 16:] = 7.0
    assert hmm.spline_penalty(model, 8).item() == 0.0
    with torch.no_grad():
        model.W[0, 0, 0] = 4.0                                   # one knot bumped by 1 -> two unit differences
    assert hmm.spline_penalty(model, 8).item() == pytest.approx(2.0)


def test_select_kan_table():
    p = ft.load_panel()
    M, lam, table = hmm.select_kan(p, {}, folds=("f4",), M_grid=(8,), lambda_grid=(0.0, 1.0), tau=10.0,
                                   half_life=60, max_epochs=2, device="cpu")
    assert table.columns == ["M", "lambda_spl", "nll_inner"] and table.height == 2 and M == 8
```

Append to `tests/test_leakage.py`:

```python
def test_b4kan_ignores_post_origin_values(panel):
    C, _ = ev.thresholds(panel, ft.role_idx(panel, "f4", "train"))
    m = hmm.hmm_model("B4", kan=True, kan_M=8, kan_lambda=0.01, **HMM_KW)
    st = m["fit"](panel, "f4", C)
    d = panel["dates"].index(D0)
    same(m["predict"](st, panel, d), m["predict"](st, perturb(panel, d, d, d), d))
```

and to `tests/test_run_all.py`:

```python
def test_loop_registry_complete():
    from gmst import run_all
    assert run_all.LOOP == ("H", "IO", "KAN")
```

- [ ] **Step 2: Run to verify they fail** — `uv run pytest tests/test_hmm.py tests/test_leakage.py tests/test_run_all.py -q` → exactly the 8 new tests fail (`no attribute 'cyclic_bspline'` …); all earlier tests pass.
- [ ] **Step 3: Implement** the KAN transition mode, the penalty, `select_kan`, `LOOP` and `kan_grid.csv`.
- [ ] **Step 4: Run to verify they pass** — same command → all passed; `uv run pytest -q` → all passed.
- [ ] **Step 5: Commit**
```bash
git add gmst/hmm.py gmst/run_all.py tests/test_hmm.py tests/test_leakage.py tests/test_run_all.py
git commit -m "feat: improvement loop 3 - B4-KAN periodic cubic B-spline transition logits with smoothness penalty"
```
- [ ] **Step 6: Dev run and gate** — as Task 21 Step 6; the loop ends here in any case → Task 24.

---

### Task 24: The single sealed final run, results tests, determinism

**Files:**
- Create: `tests/test_results.py`
- Create (by running): `results/*` (committed)

**Rules:** this is the only task that runs the final stage for real (the smoke test's final stage writes into a temp dir and is only schema-checked). The final stage evaluates the sealed window once, for the one B4 variant chosen by `select_variant` (plus B0/B0′/B1 for comparison, FR-94 ⑬). Do not open, print or summarise `results/test_metrics.csv` before Step 3 is done, and never change a model, variant or threshold because of it. If a bug fix is needed afterwards, fix it, rerun once, and record the rerun and its reason in the commit message. Target date: 2026-10-04 (report 10-04…10-06, buffer 10-07…10-08).

- [ ] **Step 1: Write the results test** — create `tests/test_results.py` (it skips until the final run exists):

```python
import json

import polars as pl
import pytest

from gmst import RESULTS
from gmst import evaluate as ev

pytestmark = pytest.mark.skipif(not (RESULTS / "selection.json").exists(), reason="final run (Task 24) not done yet")
MODELS = ("B0", "B0p", "BB", "B1", "B2", "B3", "B4")
MODEL_ID = {"main": "B4", "H": "B4-H", "IO": "B4-IO", "KAN": "B4-KAN"}


def load(name):
    return pl.read_csv(RESULTS / name, infer_schema_length=None)


def sel():
    return json.loads((RESULTS / "selection.json").read_text(encoding="utf-8"))


def test_g3_baselines_reproduced():
    assert [round(v, 2) for v in load("lag7_skip.csv")["mae"]] == [6.23, 25.77, 65.34, 9.84]
    m = load("metrics.csv").filter((pl.col("model") == "B0p") & (pl.col("variant") == "main")
                                   & (pl.col("stratum") == "all") & (pl.col("metric") == "mae")
                                   & pl.col("fold").is_in(["f1", "f2", "f3", "f4"])).sort("fold")
    assert [round(v, 2) for v in m["value"]] == [14.85, 10.38, 14.49, 15.17]


def test_g4_same_conditions_table():
    m = load("metrics.csv").filter((pl.col("variant") == "main") & (pl.col("fold") == "pooled")
                                   & (pl.col("stratum") == "all"))
    for model in MODELS:
        assert {"mae", "rmse", "peak_mae", "peak_hit2", "n_events_C90"} <= set(m.filter(pl.col("model") == model)["metric"])
    for model in ("B1", "B2", "B3", "B4"):
        assert {"crps", "cov80", "brier_platt_C90", "auc_C90", "bss_cond_C90", "f1_C90_pstar"} <= set(
            m.filter(pl.col("model") == model)["metric"])
    assert m.filter(pl.col("metric") == "mae")["n"].n_unique() == 1


def test_oof_row_counts():
    days = load("oof_days.csv")
    for model in MODELS:
        assert days.filter((pl.col("model") == model) & (pl.col("variant") == "main")).height == 56
    for v in ("K2", "K4", "re", "copies", "suspect", "protoA"):
        assert days.filter((pl.col("model") == "B4") & (pl.col("variant") == v)).height == 56, v
    for v in sel()["tried"][1:]:
        assert days.filter((pl.col("model") == "B4") & (pl.col("variant") == v)).height == 56, v
    assert days.filter((pl.col("model") == "B1") & (pl.col("variant") == "AWstar")).height == 56
    assert days.filter(pl.col("variant") == "B").height == 42
    assert load("oof_slots.csv").filter((pl.col("model") == "B4") & (pl.col("variant") == "main")).height == 5376


def test_g5_selection_consistent():
    s, boot = sel(), load("bootstrap.csv")
    run = {c["variant"]: {"mae": c["mae"], "brier_mean": c["brier_mean"], "gate_met": c["gate_met"], "gap": {}}
           for c in s["candidates"] if c["status"] == "run"}
    assert ev.select_variant(run)["submitted"] == s["submitted_variant"]
    assert s["submitted"] == MODEL_ID[s["submitted_variant"]] and s["submitted"] != "B1"
    for v, c in run.items():
        assert ev.gate_pass(boot, MODEL_ID[v] + "-B1") == c["gate_met"]
    assert {"B4-B1", "AWstar-main"} <= set(boot["comparison"]) and (boot["n_days"] == 54).all()
    r = boot.filter((pl.col("comparison") == s["submitted"] + "-B1") & (pl.col("metric") == "mae")).row(0, named=True)
    assert s["criteria"]["mae"] == pytest.approx([r["delta"], r["ci_lo"], r["ci_hi"]])


def test_g6_risk_check_recorded():
    rc = json.loads((RESULTS / "risk_check.json").read_text(encoding="utf-8"))
    for key in ("B1/main", "B2/main", "B3/main", "B4/main", "B4/re"):
        assert {"C50", "C75", "C90", "discrimination_missing"} <= set(rc[key])
        assert rc[key]["C90"]["n_events"] == 13


def test_g7_scenarios():
    d, mo = load("scenarios.csv"), load("scenarios_month.csv")
    assert d.height == 27 * 13
    assert set(d["scenario"]) == {"baseline", "shift_peak", "stagger_start", "avoid_high", "ease_peak"}
    assert mo.height == 2 * 13 and (mo["P_floor"] == 222.0).all()
    assert (mo.filter(pl.col("scenario") == "baseline")["d_demand_won"] == 0).all()


def test_report_support_files():
    s = load("state_stability.csv")
    assert s.columns == ["seed_a", "seed_b", "occ_maxdiff", "delta_maxdiff", "same_order", "same_conclusion",
                         "agreement"] and s.height == 3
    assert load("leakage_gap.csv")["scheme"].to_list() == ["random5", "loeo"]
    assert load("hmm_transitions.csv").columns == ["op", "daytype", "hour", "from_state", "p_to_high"]
    assert load("backbone_tau.csv").height == 15


def test_g1_submission_shape():
    t = load("test_predictions.csv")
    assert t.height == 1344 and len(t.columns) == 33 and set(t["model"].unique()) == {sel()["submitted"]}
    assert not (RESULTS / "test_predictions_B4_module.csv").exists()
```

- [ ] **Step 2: Single final run** — `rm -rf results && uv run python -m gmst.run_all 2>&1 | tee run_final.log` (add `--occ-floor` if Task 20 required it). Expected: exit 0, 14 stage lines, `results/test_predictions.csv` present.
- [ ] **Step 3: Verify** — `uv run pytest tests/test_results.py -q` → `8 passed`; `uv run pytest -q` → all passed.
- [ ] **Step 4: Determinism (success metric)** — `uv run python -m gmst.run_all --out results_repro` (same flags), then
  `uv run python -c "import polars as pl, numpy as np; a=pl.read_csv('results/test_predictions.csv'); b=pl.read_csv('results_repro/test_predictions.csv'); c=[k for k,t in a.schema.items() if t.is_numeric()]; print(np.abs(a.select(c).to_numpy()-b.select(c).to_numpy()).max())"` → prints a value ≤ `1e-6`; then `rm -rf results_repro`.
- [ ] **Step 5: Commit**
```bash
git add tests/test_results.py results
git commit -m "run: improvement-loop selection + single sealed final run; results committed"
```

---

### Task 25: Display-only notebooks (03 new, 01/02 re-pointed)

**Files:**
- Create: `notebooks/03_results.ipynb`, `tests/test_notebooks.py`
- Modify: `notebooks/01_preprocess.ipynb`, `notebooks/02_eda.ipynb`

**How to edit without reading base64 outputs:** never open the raw `.ipynb`. Print cell sources only with `uv run python -c "import json; [print(i, c['cell_type'], ''.join(c['source'])) for i, c in enumerate(json.load(open('notebooks/02_eda.ipynb'))['cells'])]"`; write notebooks with a throw-away `uv run --with nbformat python` script (not committed) that sets cell sources and clears outputs; then execute each notebook in place: `cd notebooks && uv run --with nbconvert jupyter nbconvert --to notebook --execute --inplace <name>.ipynb`.

**Content contract (US-019, FR-103…FR-105):**
- All three notebooks: read files only (no `write_csv`, `to_csv`, `write_json`, `write_parquet`, `.savefig(`), no `import gmst`, no `lightgbm`, no `torch`; paths are `../5. 자원 최적화 AI 데이터셋/…` and `../results/…`; Korean font setup as in the current notebooks.
- `01_preprocess.ipynb` (rewrite as a viewer): load `../5. 자원 최적화 AI 데이터셋/okm_15min_2021.csv` and `../results/data_audit.json`; structural checks (24,672 rows, 15-minute grid, 96 per day, `전력` nulls == `is_missing` count == 74); print exactly `감사: zero_points=74, wind_null_hours=3, precip_null_hours=1` from the audit values; the existing plots (15-min series + 1-day moving average, date × quarter heatmap, weekday/weekend profile, histogram with p95/p99, power vs production and temperature scatter) restricted to rows ≤ 2021-08-31. No `numpy` import (the old unused import goes away).
- `02_eda.ipynb` (re-point): right after loading the 15-min CSV, keep only rows ≤ `date(2021, 8, 31)` for every statistic and plot; read the day table `../5. 자원 최적화 AI 데이터셋/okm_cv_splits_2021.csv` instead of recomputing groups, operating flags, anomalies and fold roles (delete the cells that compute and write the split CSV; keep the fold timeline plot, now drawn from the CSV's role columns); print `비가동일 수: 62` from the day table's `is_operating` (all 257 days); masked design statistics use the day table's `is_copy`/`is_suspect` and `is_missing`. §6 R² on 07-01…08-31 masked points. §8: replace the 100-bin + 5-bin moving-average mode count with `np.histogram(values, bins=np.arange(0, 240, 10))` over 07-01…08-31 masked points and print exactly `10단위 빈 3봉: 20–30대 2082 / 100–110대 569 / 170–180대 326` from the computed counts; add a markdown line "100빈+5빈 이동평균에서 센 10개 모드는 잔물결이다(v2 §4.4)". §11: compute on ≤ 08-31 and add the two caveats as markdown (the two lag-1 rows use different operating-flag definitions; the H=96 sliding-window exceedance is not midnight-aligned and mixes operating/non-operating regimes, v2 §4.5). Replace the week plot 2021-09-06…09-12 with 2021-08-09…08-15; the cross-correlation "clean" window becomes 07-01…08-31; the ACF uses ≤ 08-31.
- `03_results.ipynb` (new; sections in report-chapter order): Ch.1 data caveats (`data_audit.json`, day-table counts, fold timeline, `leakage_gap.csv`); Ch.2 the B4-family candidate table from `selection.json["candidates"]` first (every pre-registered variant `main, H, IO, KAN` with its status, pooled MAE, mean Brier, `gate_met`, the `<model>-B1` CIs from `bootstrap.csv`, the printed line `시도한 B4 후보 수: <n_candidates_run> / 4` and a multiple-comparison caveat; `io_lambda.csv` / `kan_grid.csv` when present), then model comparison (`metrics.csv` pooled/main/all table of mae, rmse, crps, peak_mae, brier_platt_C90, auc_C90, bss_cond_C90, f1_C90_pstar with n and event counts; `bootstrap.csv`; `selection.json`; `risk_check.json`; `backbone_tau.csv`; the `AWstar` rows labelled "휴리스틱 상한"); reliability curves before/after calibration from `oof_days.csv` (risk_raw vs risk_platt, 5 bins); Ch.3 `error_by_condition.csv`, `fn_fp_days.csv`, `fn_fp_summary.csv`, `hmm_transitions.csv` heatmap; Ch.4 `scenarios_month.csv` with `d_demand_won` and `d_energy_won` side by side and the note "래칫 바닥 222(07-19 11:15) → 7–8월 기본요금 절감 ≈ 0, 주 수단은 TOU 전력량요금 이동", then `scenarios.csv` (daily J_d components); Ch.5 `state_stability.csv` and the reliability figure; Ch.6 test prediction plot from `test_predictions.csv` (y_median with the q10–q90 band) and the `eval_mask.csv` missing count.

- [ ] **Step 1: Write the failing test** — create `tests/test_notebooks.py`:

```python
import json
from pathlib import Path

import pytest

NB = Path(__file__).resolve().parents[1] / "notebooks"


def cells(name):
    return json.loads((NB / name).read_text(encoding="utf-8"))["cells"]


def code(name):
    return "\n".join("".join(c["source"]) for c in cells(name) if c["cell_type"] == "code")


def outputs(name):
    out = []
    for c in cells(name):
        for o in c.get("outputs", []):
            out += o.get("text", []) + o.get("data", {}).get("text/plain", [])
    return "".join(out)


@pytest.mark.parametrize("name", ["01_preprocess.ipynb", "02_eda.ipynb", "03_results.ipynb"])
def test_viewer_only(name):
    src = code(name)
    for bad in ("write_csv", "to_csv", "write_json", "write_parquet", ".savefig(", "import gmst", "from gmst",
                "lightgbm", "torch"):
        assert bad not in src, (name, bad)
    assert all(c.get("execution_count") for c in cells(name) if c["cell_type"] == "code"), name


def test_01_reads_script_output():
    src = code("01_preprocess.ipynb")
    assert "okm_15min_2021.csv" in src and "data_audit.json" in src and "okm_augumented_2021.csv" not in src
    assert "import numpy" not in src
    assert "감사: zero_points=74, wind_null_hours=3, precip_null_hours=1" in outputs("01_preprocess.ipynb")


def test_02_design_stats_sealed():
    src = code("02_eda.ipynb")
    assert "okm_cv_splits_2021.csv" in src and "date(2021, 8, 31)" in src
    assert "date(2021, 9, 6)" not in src and "date(2021, 9, 14)" not in src
    out = outputs("02_eda.ipynb")
    assert "비가동일 수: 62" in out
    assert "10단위 빈 3봉: 20–30대 2082 / 100–110대 569 / 170–180대 326" in out


def test_03_reads_results():
    src = code("03_results.ipynb")
    for f in ("metrics.csv", "bootstrap.csv", "selection.json", "risk_check.json", "oof_days.csv",
              "error_by_condition.csv", "fn_fp_days.csv", "scenarios_month.csv", "scenarios.csv",
              "state_stability.csv", "leakage_gap.csv", "test_predictions.csv", "eval_mask.csv"):
        assert f in src, f
```

- [ ] **Step 2: Run to verify it fails** — `uv run pytest tests/test_notebooks.py -q` → failures (`03_results.ipynb` missing; `01` still writes a CSV).
- [ ] **Step 3: Implement** the three notebooks per the contract and execute them in place with nbconvert (command above).
- [ ] **Step 4: Run to verify it passes** — `uv run pytest tests/test_notebooks.py -q` → `6 passed`.
- [ ] **Step 5: Full suite and commit**
```bash
git add notebooks/01_preprocess.ipynb notebooks/02_eda.ipynb notebooks/03_results.ipynb tests/test_notebooks.py
git commit -m "feat: display-only notebooks (03 results viewer; 01/02 re-pointed to script outputs, design stats <= 08-31)"
```

---

### Task 26: README, PONYTAIL-DEBT.md ledger, final verification, package

**Files:**
- Modify: `README.md` (currently empty), `tests/test_style.py` (append)
- Create: `PONYTAIL-DEBT.md` (at `WT` root, i.e. the repository root after merge)

**Contract:**
- `README.md` (Korean, FR-101, no affiliation/logo, FR-7), sections in order: 개요 (문제정의 H=96 / L=672 / 00:00 발행 / C = fold 학습 가동일 일최대 q50·q75·q90, 테스트 186.5·198·210.7); 환경 (`uv sync` 또는 `pip install -r requirements.txt`, torch cu126 index, `CUDA_VISIBLE_DEVICES=""`로 CPU 실행); 단일 명령 (`uv run python -m gmst.run_all`, `--no-final`, `--smoke`, `--occ-floor`, `--package`); 산출물 표 (file → 보고서 장, PRD §6 table); 데이터 주의점 요약 (DC1–DC16 처리 규칙 한 줄씩); 모델 선택 (B1은 벤치마크, 제출은 항상 B4 계열; 게이트 = B1 대비 ΔMAE·Δ평균Brier CI 상한 < 0; 개선 루프 B4-H → B4-IO → B4-KAN과 `selection.json`의 시도 목록·개수·다중비교 주의문, B1 대비 잔여 격차); 제출파일 스키마 (`test_predictions.csv` 33열, `model` 열 의미 = `select_variant`가 정한 B4 변형 ID `B4`/`B4-H`/`B4-IO`/`B4-KAN`, `eval_mask.csv`); 외부자료 (`HOLIDAYS_2021` 하드코딩과 출처 한국천문연구원 특일정보, 한전 산업용(을) 2021 요금표와 출처, 외부 기상 미사용과 근거 = oracle-weather `AWstar − main` 결과(휴리스틱 상한, 수치는 `results/bootstrap.csv`에서 인용)); ₩ 가정 (전력 단위 "15분 평균 kW" 가정, 상대변화(%) 병기, 래칫 바닥 222 → 7–8월 기본요금 절감 ≈ 0, kWh당 가산요금은 `d_kwh ≈ 0`일 때만 상쇄); 재현성 (seed 0, 날짜별 MC seed, CUDA/CPU, 같은 장치 재실행 차 ≤ 1e-6); 폴더 구조; 제출 zip (`dist/kamp_power_src.zip`).
- `PONYTAIL-DEBT.md`: generated with the `ponytail:ponytail-debt` skill (invoke it with the Skill tool), then shaped into per-module Markdown tables with the header `| file:line | 지름길 | 한계 (ceiling) | 업그레이드 조건 (trigger) | 근거 |`; one row per line found by `grep -rnE '(#|//) ?ponytail:' gmst tests`, the first cell is `gmst/<file>.py:<line>` (or `tests/…`); the 근거 cell carries the FR / v2 / 수정 reference from the marker. Add a "발동된 트리거" section listing any trigger that fired in Tasks 20–24 (e.g. B1 ≥ B0′, occupancy collapse), or "없음".

- [ ] **Step 1: Write the failing tests** — append to `tests/test_style.py`:

```python
HARVEST = re.compile(r"(#|//) ?ponytail:")
LEDGER_ROW = re.compile(r"^\|\s*`?(gmst|tests)/[^|]+:\d+`?\s*\|")


def test_debt_ledger_complete():
    markers = [f"{p.relative_to(ROOT).as_posix()}:{i}" for p in _py("gmst", "tests")
               for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1) if HARVEST.search(line)]
    rows = [l for l in (ROOT / "PONYTAIL-DEBT.md").read_text(encoding="utf-8").splitlines() if LEDGER_ROW.match(l)]
    assert markers and len(rows) == len(markers)
    joined = "\n".join(rows)
    assert all(m in joined for m in markers)


def test_readme_sections():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for s in ("uv run python -m gmst.run_all", "pip install -r requirements.txt", "uv sync", "test_predictions.csv",
              "eval_mask.csv", "oracle", "15분 평균 kW", "HOLIDAYS_2021", "kamp_power_src.zip",
              "CUDA_VISIBLE_DEVICES", "--no-final", "222"):
        assert s in text, s
```

- [ ] **Step 2: Run to verify they fail** — `uv run pytest tests/test_style.py -q` → `FileNotFoundError: … PONYTAIL-DEBT.md` and README assertion errors.
- [ ] **Step 3: Write** `README.md` (quote numbers from `results/`, never from memory) and `PONYTAIL-DEBT.md` (skill + shaping). Expected markers at minimum: `splits.py` (holidays), `baselines.py` (LightGBM defaults), `evaluate.py` (day-block bootstrap), `hmm.py` (NaN production in z), `baselines.py` (B4-H cross-fitted centre, if Task 21 ran), `scenario.py` (ratchet assumptions; path pairing), `analysis.py` (L2-only leakage gap), `run_all.py` (requirements fallback).
- [ ] **Step 4: Final verification**
  - `uv run pytest -q` → all passed; `CUDA_VISIBLE_DEVICES="" uv run pytest -q` → all passed (G2).
  - `grep -rnE '(#|//) ?ponytail:' gmst tests | wc -l` equals the ledger row count (the test checks it).
  - `uv run python -m gmst.run_all --package` → `dist/kamp_power_src.zip` (not committed; `dist/` is ignored).
- [ ] **Step 5: Commit**
```bash
git add README.md PONYTAIL-DEBT.md tests/test_style.py
git commit -m "docs: README (FR-101) and PONYTAIL-DEBT.md ledger harvested from ponytail markers"
```

---

## Coverage (PRD → tasks)

**User stories** (19 active; US-013/US-014 deleted by the KMA decision → no task; spec frozen at a82ea9d):

| Story | Task(s) |
|---|---|
| US-001 deps & skeleton | 1 |
| US-002 preprocessing | 2 |
| US-003 day table & roles | 3 |
| US-004 loader, protocols, seal, `lgbm_rows` | 4, 8 |
| US-005 metrics, rolling origin, naive baselines | 5, 6 |
| US-006 backbone | 7 |
| US-007 B1 + oracle-weather | 8, 10, 18 |
| US-008 calibration, discrimination, bootstrap, `gate_pass` | 9, 10, 18 |
| US-009 conditional HMM core | 11 |
| US-010 MC risk, ladder, K, `re` diagnostic | 12, 18 |
| US-011 P0-4 leakage test | 13 (+ additions in 21, 22, 23) |
| US-012 P1-3 three-seed stability | 14 |
| US-015 tariff | 15 |
| US-016 SCENARIO what-ifs | 16 |
| US-017 error analysis | 17 |
| US-018 runner & submission | 18, 19, 24 |
| US-019 notebooks | 25 |
| US-020 ponytail ledger | 26 |
| US-021 pre-registered improvement loop (B4-H, B4-IO, B4-KAN; `select_variant`) | 10, 18, 20, 21, 22, 23, 24 |

**Functional requirements** (104 active; FR-76…FR-80 and FR-98 deleted):

| FR | Task(s) | FR | Task(s) | FR | Task(s) |
|---|---|---|---|---|---|
| 1–2 | 1 | 38 | 5, 6 | 69–74 | 12 |
| 3 | 1, 2, 3, 18 | 39 | 5 | 75 | 12, 18 (diagnostic `re`) |
| 4 | 11 | 40–41 | 10 | 81 | 8, 10, 18 |
| 5 | 1, 8, 11, 12 | 42 | 9, 10 | 82–84 | 15 |
| 6 | 1, 26 | 43 | 6 | 85 | 15, 16 |
| 7 | 26 | 44 | 6, 10, 18 | 86 | 16 |
| 8–15 | 2 | 45–48 | 6 | 87 | 12, 16 |
| 16–26 | 3 | 49–50 | 8 | 88–89 | 16 |
| 27–30 | 4 | 51 | 7 | 90–92 | 17 |
| 31–32 | 8 | 52–55 | 9 | 93 | 14, 17 |
| 33 | 13 (+ 7, 9, 12 for R10) | 56 | 10 | 94–97, 99 | 18 (98 deleted) |
| 34 | 6 | 57 | 10, 18 (as amended) | 100–102 | 19 |
| 35–37 | 5 | 58–62 | 7 | 103–105 | 25 |
| 107 | 21 | 63–68 | 11 | 106 | 26 |
| 108 | 22 | 109 | 23 | 110 | 10, 18, 20–24 |

**Data caveats:** DC1 → 3, 4 · DC2 → 3, 4, 8 · DC3 → 2, 5, 6 · DC4 → 2 · DC5 → 2 · DC6 → 2 · DC7 → 4, 15 · DC8 → 8, 13, 16 · DC9 → 3 · DC10 → 4, 13, 18, 24 · DC11 → 15 · DC12 → 3 · DC13 → 3 · DC14 → 3, 4 · DC15 → 2 · DC16 → 11.

**Goals:** G1 → 18, 24 · G2 → every task, 26 · G3 → 6, 7, 24 · G4 → 6, 10, 18, 24 · G5 (as amended: gate + B4-family selection) → 10, 18, 24 · G6 → 10, 24 · G7 → 16, 24 · G8 → 26.

Result: 19/19 active stories, 104/104 active FRs, 16/16 caveats and 8/8 goals map to at least one task.

## Self-review notes

- Placeholder scan: no "TBD/TODO/similar to"; every code step has complete test code; implementation code is intentionally absent (user override).
- Name consistency checked across tasks: `rolling_origin` returns 4 values everywhere; state keys (`model, m, C, s_op, best_epoch, history, val_nll, device, protocol, kind, K, train_idx, inner_state`); loop names (variants `main/H/IO/KAN`, model IDs `B4/B4-H/B4-IO/B4-KAN`, comparisons `<model ID>-B1`, factory flags `emission_source`, `decoder`, `kan`); `gate_pass` / `select_variant` (spec a82ea9d names); file schemas `SLOT_COLS`, `DAY_COLS`, `SCN_DAY_COLS`, `SCN_MONTH_COLS`, `PRED_COLS`.
- Numbers asserted in tests were re-derived read-only on data ≤ 2021-08-31 while writing the plan (copy rule and partners, usable counts 6,240 / 11,256 / 11,352 / 23,064 / 11,544 / 23,256, thresholds and event counts per fold, lag-7 and B0′ tables, backbone R²/sd/lag-1, ratchet 222 at 07-19 11:15, natural same-slot maximum 9, 10-unit bins 2082/569/326, inner-window and scenario-day counts). Values involving 2021-09-01…09-14 (74 zeros incl. 09-08, 142 events, C_test from ≤ 08-30 only) come from the spec, not from computation.

## Spec issues found

1. **FR-56 / v2 A13 "54 usable OOF days"** — the 54-day set (56 minus the suspect days) still contains 2021-07-30, an exact copy whose target is fully masked, so only 53 days carry weight. The plan implements the ruled 54 (R5) and asserts it from the day table; 07-30 contributes zero to every numerator and denominator.
2. **US-006 lag-1 and "01-02를 빼면 4.47"** — 0.909 / 0.961 reproduce only when NaNs are dropped and consecutive observed residuals are paired (the adjacent-pair definition gives 0.962 on ≤ 08-31); 4.47 is the non-op sd with 01-02's residuals dropped but the cell means kept (refitting without 01-02 gives 1.68). The plan pins both definitions in its tests.
3. **US-005 thresholds** — f3 C75 is exactly 197.75 (the PRD prints 197.8); tests use 197.75.
4. **DC11 / v2 부록 B monthly mid/peak maxima** (Feb 195, Mar 200, Jun 202) include copy days; under FR-85's masking Feb is 193 and Mar/Jun have no usable mid/peak points. The 222 floor (07-19 11:15) is unaffected.
5. **FR-33 vs FR-54** — FR-33 lists 보정함수 among the inner-validated items while FR-54 keeps leave-one-fold-out calibration; the plan follows FR-54 (it also never uses a fold's own validation labels).
6. **R10 inner windows overlap earlier validation windows** — the inner windows of f2–f4 (07-13…19, 07-27…08-02, 08-10…16) lie inside the f1–f3 validation windows, so any pooled-over-folds inner selection (FR-61 (τ, h); the IO/KAN grids) touches earlier folds' validation labels. The plan keeps FR-61's pooling but makes p* strictly per fold (own inner window; test p* from 08-24…08-30).
7. **FR-61 test-fit parenthetical** "(≤08-30 학습구간의 마지막 7일 내부검증)" is ambiguous; the plan uses one global (τ, h) everywhere.
8. **Daily η\*_d** — v2 §12 reports a daily break-even, but PRD US-016's `scenarios.csv` schema lacks it; the plan appends `eta_star` (v2 wins).
9. **FR-102 vs v2 §20 ②** — the zip list omits `PONYTAIL-DEBT.md`; the plan includes it (v2 wins).
10. **US-012** asks for tolerance checks yet says to report "seed 의존" when exceeded; the plan gates the tolerances on synthetic data and records the real-data verdict in `state_stability.csv`.
11. **FR-38** is silent on recomputing the issued risk on observed slots; the plan keeps the issued risk (only 09-08 is affected).
12. **B4-H training-day centre (FR-107, A34)** — "fold-safe B1 forecast" is defined for issue days only; B1's own fit on the fold's train window would give in-sample centres on the HMM's training days. The plan pins 5-block cross-fitting inside the train window for those days.
13. **`emission_source` placement (US-021)** — the PRD puts it on the `CondHMM` constructor, but `m_t` is data built outside the torch module; the plan puts the switch on the `hmm_model` factory and tests the PRD property (`m_t` equals `predict_b1`'s median on issue days).
14. **`gate_pass` signature (US-008)** — the PRD writes `gate_pass(variant_metrics, b1_metrics)`; the plan passes the bootstrap rows (`gate_pass(boot, "<model>-B1")`), which already hold both sides.
15. **B4-KAN (FR-109, v2 §9.6 ③)** — the formula keeps an intercept `b_ij` next to per-op splines, while the stated parameter count (2M + 4 per pair) has none, and a partition-of-unity spline makes the intercept redundant; the plan drops it. The 4-value λ_spl grid ("격자 3×4") is not listed; the plan uses {0, 0.01, 0.1, 1}, the IO grid, with pooled inner NLL as the criterion for both.
16. **FR-99 smoke** necessarily runs the final stage on the test window (into a temp dir); the plan forbids reading those values and only schema-checks them.
