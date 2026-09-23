# KAMP Power Forecast (GMST-Power Contest Core v1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Current execution decision:** Astra implements the code with Ponytail and the reviewed contracts below. The old fable/commit instructions are superseded. The plan contains acceptance contracts; implementers write the minimum runnable checks, run the relevant tests, and do not commit or push unless separately asked.

**Goal:** Development command (uv run python -m gmst.run_all) builds repaired data, OOF comparison, calibration, exploratory B4-vs-B1 gate, scenarios and analysis while keeping September sealed. Smoke exercises Aug 18–31 pseudo-test. Only the explicit final command (uv run python -m gmst.run_all --final), after selection is frozen, writes the 1,344-row September submission and test metrics.

**Architecture:** Keep the flat gmst package and function-based models. The numerical HMM may be split into flat hmm_core.py, training.py, forecast.py and states.py, with hmm.py as a thin public facade. The B1 row builder and hybrid centre may similarly live in flat lgbm_features.py and hybrid.py, reexported by features.py and baselines.py. No registry, hierarchy or framework. The runner may use flat run_development.py and run_outputs.py for orchestration and file writing if needed; run_all.py remains the public entry and the sole real-unseal call site. Retain every original API from commit 6619558 unless this reviewed plan explicitly changes its signature or return contract.

**Tech Stack:** Python ≥3.13, uv, polars, numpy, torch 2.14.0+cu126 (cuda if available else cpu), lightgbm (native `lgb.train` API), matplotlib (notebooks), pytest (dev).

**Spec:** PRD US-001…US-021 (19 active), FR-1…FR-110 (104 active), DC1…DC16; proposal v2 A1…A36; approved review dated 2026-09-23; later decisions in .sdd/progress.md. The review corrections in this plan supersede conflicting historical change-log rows.

## Global Constraints

- Work only inside the worktree `WT = /home/user/manufacture_ai/.claude/worktrees/kamp-core`, branch `kamp-core`. Run every command from `WT` with `uv run …`. The first `uv run` creates `WT/.venv` from the uv cache.
- Git operations: no commits, staging, pushing, or PR creation in this implementation request. Keep the worktree reviewable.
- Dependencies: only `uv add lightgbm` (runtime) and `uv add --dev pytest` (dev). Never add or import `scipy`, `sklearn`/`scikit-learn`, `pandas`, `requests`. Keep the torch `[tool.uv.sources]` / `[[tool.uv.index]]` (pytorch-cu126) blocks unchanged (FR-1, FR-2).
- LightGBM: native API only (`lightgbm.Dataset`, `lightgbm.train`); the sklearn wrapper needs scikit-learn and is forbidden. Params = library defaults + `{"seed": 0, "deterministic": True, "force_col_wise": True, "verbose": -1}`, `num_boost_round=100` unless a smoke/test argument lowers it (FR-5, FR-49).
- ponytail (binding style): the laziest code that works; stdlib → already-installed deps (polars, numpy, torch, matplotlib) first; no class/abstraction without a second use (the model dict protocol has 7+ uses; `CondHMM` is the only class); fewest files; no config files (constants at the top of the module that uses them). Every deliberate shortcut carries `# ponytail: <ceiling>, <upgrade trigger> (<FR/v2 ref>)` on one line; the marker must match the regex `# ?ponytail:\s*[^,]+,\s*\S+` so that `grep -rnE '(#|//) ?ponytail:' gmst tests` finds it (FR-6).
- Seal (FR-28, DC10, review R1): load_panel hides every 09-01…09-14 Y and X unless unseal=True. No development test, smoke, or reproducibility command invokes real unseal. Default runner and --no-final stop after development stages. --smoke runs a pseudo-final with f4 train and 08-18…08-31 as its 14-day pseudo-test; it uses load_panel() with the September seal intact. --final is the sole explicit real-unseal path, allowed only after selection.json fixes model, variant, parameters, and p*. --smoke and --final are mutually exclusive. Task 24 invokes --final once; later rerun only to repair a documented execution defect, never to reselect a model.
- B1 is a benchmark only: only B4 family is submitted. The MAE and mean-Brier CI gate uses the derived 53 usable OOF days and is an exploratory development gate after candidate selection, not a general superiority claim. The fixed B1 default has 100 rounds; no new deep encoder.
- Masking is applied in three places at once: targets, lag/rolling inputs, HMM warm-up (DC1, DC2, DC14, [수정-5]). The loader does it once by writing NaN into `Y` (and `생산량` for suspect days); downstream code must treat NaN as "unknown" and never fill it.
- Internal split (FR-33/54/55/61/68, review R2/R3): for each fold and variant, order its train-role days with ≥1 finite target. With n such days, reserve ≥5 for initial fit; cal_count=min(7,max(0,n−10)) newest days; tune_count=min(7,max(0,n−cal_count−5)) preceding days. features.internal_split(panel, fold, train_idx=None) returns (fit_idx,tune_idx,cal_idx): fit_idx contains all train dates before tune_idx (or cal_idx if tune empty), including dates without usable target; tune_idx and cal_idx contain usable days only. features.tuning_idx and features.inner_idx return indices 1 and 2 respectively. Tune hyperparameters/epochs on tune_idx using fit_idx, fit a calibration predictor on all train dates before cal_idx, predict cal_idx forward, then refit on the full train for outer validation. Empty tuning uses fixed defaults with a status/count; empty calibration uses identity probability calibration and p*=0.5 with a status/count. Never pool other folds into fold j settings. For final September inference, repeat selection on the final pre-September train and fit the final calibrator from same-(model,variant) f1–f4 pre-September OOF.
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
| gmst/features.py; optional flat lgbm_features.py | panel loader, seal, protocols, calendar, B1 row builders; features.py retains public names | 4, 8 |
| `gmst/evaluate.py` | metrics, thresholds, rolling-origin loop, calibration, p*, climatology, risk check, bootstrap, DM, selection | 5, 6, 9, 10 |
| gmst/baselines.py; optional flat hybrid.py | B0/B0′/B1 public facade and H hybrid centre; baselines.py retains public names | 6, 8, 21 |
| `gmst/backbone.py` | cyclic-RW2 backbone, BB model, (τ,h) selection, as-of backbone | 7 |
| gmst/hmm.py (public facade); optional flat hmm_core.py, training.py, forecast.py, states.py | conditional HMM numerical work and public API; split only if needed for clear ownership, no registry/framework | 11, 12, 14, 21–23 |
| `gmst/scenario.py` | KEPCO 2021 tariff, ratchet, schedule transforms, SCENARIO what-ifs | 15, 16 |
| `gmst/analysis.py` | error by condition, FN/FP tables, augmentation-leakage gap | 17 |
| gmst/run_all.py; optional flat run_development.py and run_outputs.py | public CLI and sole unseal call; helpers may orchestrate development/write outputs | 18, 19 |
| `tests/test_style.py` | ponytail marker format, banned imports, unseal location, deps, ledger/README checks | 1, 22 |
| `tests/test_preprocess.py`, `test_splits.py`, `test_features.py`, `test_evaluate.py`, `test_baselines.py`, `test_backbone.py`, `test_hmm.py`, `test_leakage.py`, `test_state_stability.py`, `test_scenario.py`, `test_analysis.py`, `test_run_all.py`, `test_results.py`, `test_notebooks.py` | tests per module / cross-cutting | 2–25 |
| `notebooks/01_preprocess.ipynb`, `02_eda.ipynb` (modify), `03_results.ipynb` (create) | display-only viewers | 21 |
| `requirements.txt`, `.gitignore` (modify), `README.md` (fill), `PONYTAIL-DEBT.md` | submission artifacts | 19, 26 |
| `results/` | written only by `run_all` (dev runs Tasks 20–23, committed after the final run in Task 24) | 20–24 |

`main.py` stays untouched (PRD non-goal).

## Shared contracts (read before any task)

**Implementation-review clarifications (2026-09-23):** Evaluation rows recompute path-model risk from the same observed-slot path maxima used for the event; point baselines likewise use observed-slot point maxima. Issued/submission risk remains the full-day prediction. B1 retains its separately trained daily classifier probability (its labels are observed-slot maxima); it has no joint path distribution from which to recompute a day-specific masked probability, and this limitation must be stated. H centre cross-fitting and the LOEO augmentation diagnostic mask held-out target values in every target-derived training feature, including lags and fitted backbone. The LOEO diagnostic also masks held-out historical targets for validation features; it is a strict event-isolation diagnostic, not the operational rolling-origin estimand. The unused legacy fit_with_inner helper is superseded by hmm_model's separate tuning/calibration/full-fit stages.

**Panel dict** (returned by `features.load_panel`, consumed everywhere). Model code may read only these keys: `dates` (list of 257 `datetime.date`, 2021-01-01…2021-09-14), `Y` (257×96 float64, NaN = masked/missing/sealed), `X` (dict with keys `생산량`, `기온`, `풍속`, `습도`, `강수량_증분`, each 257×96 float64, hourly value repeated over its 4 slots, NaN when sealed; `생산량` NaN on suspect days), `is_missing` (257×96 bool, the outage flag), `days` (polars DataFrame = the 17-column day table, `date` as `pl.Date`), `op` (257 int8, true `is_operating`), `hol` (257 int8), `dtype` (257 int8: 0 `wk`, 1 `sat`, 2 `sun`), `dow` (257 int8, Monday = 0), `month` (257 int8).

**Model dict protocol** (rolling-origin input): model dict has name, fit(panel,fold,C)->state, predict(state,panel,d)->prediction, and required inner_state(state)->state|None extractor. B0/B0p return the same no-fit state; BB/B1/HMM return their calibration fit state. Optional posthoc(state,panel,d) and evaluate_nll(state,panel,indices)->(value,n) callbacks are model-specific. The evaluator calls evaluate_nll only after predictions, never inside fit. The unchanged public APIs from commit 6619558 remain binding unless explicitly replaced in this plan. predict may use only Y[:d], X[:d], calendar(d), and protocol-allowed day-d exogenous inputs.

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

- [ ] **Step 5: Full suite and verify**

Run: `uv run pytest -q` → all passed.

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
- [ ] **Step 5: Full suite and verify** — `uv run pytest -q`, then
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
- [ ] **Step 5: Full suite and verify**

---

### Task 4: Loader, masking, protocols, seal and internal split

Files: gmst/features.py and tests/test_features.py.

Implement load_panel(include_copies=False, include_suspect=False, unseal=False), role_idx, cal_flags, internal_split(panel, fold, train_idx=None), tuning_idx and inner_idx. The loader masks copy/suspect/outage targets and lag inputs once. It hides all September Y and X by default. Only the explicit Task 24 --final branch passes unseal=True in production.

Internal split follows the global contract above. The optional train_idx restricts a variant such as SCENARIO B before the split; n counts dates with at least one finite target. Return three ascending int arrays. On n<5 return all available train dates as fit and empty tune/cal arrays; mark downstream defaults. No split member may be after that fold's train end.

Acceptance: verify known pre-September finite counts 11,352 total, 11,256 through Aug 30, default sealed September arrays all NaN, and protocol permissions. Test unseal behavior only on a synthetic temporary panel/CSV fixture; never call real load_panel(unseal=True) in pytest. Verify f1 and July-only scenario split keep at least five fit days and that no tune/cal day is in its fit. Full pytest runs with the real seal intact.

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
- [ ] **Step 5: Full suite and verify**

---

### Task 6: Naive baselines, rolling-origin loop and inner predictions

Files: gmst/baselines.py, gmst/evaluate.py, tests/test_baselines.py, tests/test_evaluate.py.

Keep model dictionaries with name, fit, predict, optional posthoc. rolling_origin(model,panel,folds=FOLDS_CV,variant="main",states=None) returns (slots,days,states,inner_days). Each model fit uses only its fold train. The full state predicts ordered outer days; model["inner_state"](state) predicts every cal_idx day without training on those days. B0 and B0p also provide the required inner_state extractor (same no-fit state), BB provides a profile fit before cal_idx. inner_days must contain actual predictions for every (model,variant,fold), including the point baselines and newly added candidates.

Use one shared day-row builder for outer and inner so risk_raw, event thresholds, and usable_peak agree. Preserve slot/day CSV schemas from the PRD. If cal_idx is empty, return a typed empty inner frame and create a later explicit fallback p* key; never silently omit the key.

Acceptance: synthetic fold with labels changed only in its outer period leaves fit state and its first 00:00 prediction invariant; changing later outer observations may change subsequent predictions. Verify B0/B0p/BB have inner rows and no outer target is used in fitting. Preserve the existing B0p 28-day and B0 lag-7 behavior.

---

### Task 7: Cyclic RW2 backbone and fold-specific (τ,h)

Files: gmst/backbone.py and tests/test_backbone.py.

Retain fit_backbone, predict_backbone, asof_matrix, fold_backbone and bb_model. select_tau(panel,fold,train_idx=None) returns (tau,half_life,table) for exactly that fold; table has fold,tau,half_life,mae_tune,n_tune,status for the 15 grid pairs. Tune on tuning_idx using only fit_idx. If n_tune=0, choose the predeclared (10.0,60) with status=default_empty. Fit the issued fold model on all train dates after choice. For the real test fold, choose anew from pre-September training dates; never reuse a pooled f1–f4 winner. B1 as-of features and all dependent models receive the fold's chosen pair.

Acceptance: perturb any outer validation labels and assert that fold's selected pair, fit state and first-day prediction do not change. A table for f1/f2 records separate selections. Warn when the chosen grid value is at an edge.

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
  - baselines.b1_model(tau,half_life,protocol="A+",rounds=100) returns a fold-fitted B1 with inner_state trained on all train dates strictly before cal_idx; the full state is trained on all train dates. It predicts every cal_idx day without including that day in fit. B1 uses the fold-specific select_tau result. Keep native LightGBM, 100-round default, 19 slot quantiles, day peak quantiles and event classifiers.

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
- [ ] **Step 5: Full suite and verify**

---

### Task 9: Platt, optional PAV, temporal calibration and p*

Files: gmst/evaluate.py and tests/test_evaluate.py.

Keep platt_fit/apply and optional pav_fit/apply. calibrate_oof(days,method="platt",inner=None,ref=None) groups by (model,variant,fold). For CV rows, fit the calibrator only on that group's train-period inner_days; apply it to that outer fold and its inner rows. For final/pseudo-test rows, ref is same-(model,variant) pre-test OOF; fit once on ref and apply to the final rows and final cal rows. Never fit a CV calibrator on other outer folds. On empty calibration rows return identity and record status=default_empty; one-class rows use the ridge Platt fit, recording status=single_class and count.

p_star_table(inner,expected_keys) returns one C50/C75/C90 entry for every requested (model,variant,fold). Use each group's actual calibrated cal-day predictions and events to maximize F1, ties deterministic. No usable cal events means p*=0.5 with status=default_empty; a single event class uses the specified deterministic p_star rule. Recompute the table after each new candidate, including B0/B0p/BB and all B4 variants. Store calibration status and n_cal in the existing result metadata.

Acceptance: perturb a fold's outer labels and assert its calibrator and p* unchanged. Perturb a later fold and assert an earlier fold's calibrated probabilities unchanged. Test H/IO/KAN same-variant reference lookup and complete expected keys.

---

### Task 10: Risk metrics, bootstrap and B4 selection

Files: gmst/evaluate.py and tests/test_evaluate.py.

bootstrap_days(panel) derives f1–f4 validation dates with ≥1 finite target; current data yields 53 (07-13, 07-15, 07-30 excluded; 08-28/29 included). Point and day-risk metrics use their own valid denominators. compare uses those dates and records n_days=53 for a full run. The 2,000-replicate day bootstrap preserves within-day dependence. Check serial correlation in day losses; if present, report a 7-day moving-block sensitivity CI. Label the simple normal day-loss test as DM-like unless autocorrelation-robust variance is implemented.

risk_metrics requires a p* entry for each (model,variant,fold) it evaluates. Isotonic metrics exist only if isotonic ran; the status table records not_run otherwise. gate_pass requires both MAE and mean Platt Brier CI upper bounds below zero versus the fixed B1. select_variant returns a B4-family variant, using OOF MAE then mean Brier for ties, and reports every attempted candidate. All CIs are exploratory after trying multiple variants, with attempted count and caveat in the report.

Acceptance: assert derived 53 days and distinct valid denominators; missing p* key raises; no iso and no K4 runs pass when marked not_run; selecting H preserves model=B4, variant=H and submitted ID B4-H.

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
  - forward_logp(model,y,obs,m,opt,z) returns scaled log contributions and filtered state probabilities. Pairwise AR emission uses the previous observed slot when available. If the immediate predecessor is missing, restart from the state stationary variance as an explicit approximation, not exact marginalization; the last observed residual is not propagated across the gap. A one-state check compares its result to the analytic one-gap Y3|Y1 mean φ²Y1 and variance σ²(1+φ²).
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
- [ ] **Step 5: Full suite and verify**

---

### Task 12: MC risk, model ladder and evaluation NLL

Files: gmst/hmm.py and tests/test_hmm.py.

Keep the B2/B3/B4 factory, common-random-number MC paths, analytic φ=0 risk, K2/K3 and diagnostic re model behavior from PRD FR-69–75. K4 is optional; when cut, record not_run and do not require its OOF rows. hmm_model(kind,protocol,K,tau,half_life,...) produces a fit state with model,m,C,s_op,best_epoch,history,device,protocol,kind,K,train_idx,inner_state. It does not read outer validation targets or store val_nll in fit. Its inner_state extractor returns a calibration state fitted on train dates before cal_idx and predicts cal_idx; tuning/early stopping uses only tune_idx and fit_idx; full model refits on all train for outer prediction. If tune_idx is empty, use a predeclared fixed epoch count (20 in smoke, 100 in full) and status=default_empty. The test fold uses final pre-September selection and full fit.

Outer NLL is computed by evaluate in the metrics stage, after prediction, from the fixed model and outer observations. Perturbing outer targets cannot change the fit state. Missing predecessor AR behavior remains the documented stationary-reset approximation; add the one-state one-gap analytic comparison check. Compare daily risk dispersion, AUC, conditional Brier skill and peak quantile coverage for B4/IO/re before claiming peak-risk improvement.

Acceptance: fixed-seed MC repeats, 96-slot path and monotone risk schemas hold; outer-label perturbation leaves fit state and first-day issued prediction unchanged; NLL appears only in evaluation output; cut K4 is not_run.

---

### Task 13: Leakage and seal regression

Files: tests/test_leakage.py.

For B0/B0p/BB/B1/B2/B3/B4 and SCENARIO, fit with fold j train, perturb j outer labels and future forbidden X, and assert unchanged j selected settings, fit parameters, calibrator, p*, and first outer day 00:00 prediction. Later outer days may legitimately use prior observed outer days, so do not assert all days invariant. Test A+ against same-day production/weather perturbations, B against future production perturbations, and AWstar against same-day weather as an oracle ablation. Repeat with H/IO/KAN when added. Test default loader keeps September Y/X NaN and that no test or smoke path calls real unseal.

Acceptance: synthetic calibration rows from a later fold do not affect an earlier fold's calibrated probabilities; selected H uses only H train-period reference. Run CPU and available GPU checks.

---

### Task 14: Three-seed state stability and transition table

Files: gmst/hmm.py and tests/test_state_stability.py.

Keep state_summary, stability_rows, transition_table and state_stability(panel,fold="f1",seeds=(0,1,2),...). Each seed repeats that fold's internal_split selection/fit/calibration procedure without outer labels in fit. Report state order, occupancy, δ and prediction agreement; real-data instability is a reported finding, while synthetic ordering checks are tests. Transition table uses the fixed trained model. If occupancy collapse is observed, the runner may rerun development with --occ-floor before final selection. Acceptance: outer-label perturbation changes neither state fit nor three-seed summaries that depend on train; synthetic known-order states pass.

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
- [ ] **Step 5: Full suite and verify**

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
- [ ] **Step 5: Full suite and verify**

---

### Task 17: Error analysis and leakage gap

Files: gmst/analysis.py and tests/test_analysis.py.

error_by_condition(slots,panel,models,variants=None,edges=None) and fn_fp(days,panel,p_star,models=None,variants=None) select explicit (model,variant) pairs. Default may be main for a standalone call, but the runner passes the submitted pair, B4/main and B1/main; do not filter the selected variant back to main. Both outputs retain model and variant columns, and p* lookup uses the row's actual variant. Display ID uses the runner's existing MODEL_ID mapping. Keep existing bins, conditions, same-slot leakage gap and transition table behavior.

Acceptance: synthetic H selection produces H rows in error_by_condition and FN/FP, with H p* values, while B1/main comparison remains present. Perturb post-origin data to confirm issued predictions do not change.

---

### Task 18: Runner, smoke pseudo-test, selection and submission

Files: gmst/run_all.py and tests/test_run_all.py.

main(argv=None) accepts --smoke, --no-final, --final, --out, --occ-floor, --package. Default and --no-final run development stages through selection/scenarios only. --smoke runs a complete schema exercise using f4 as a pseudo-test on Aug 18–31 with the normal sealed loader and reduced epochs/paths/bootstrap; --final loads and validates an already frozen selection.json, runs only the real September inference/evaluation path, and unseals once; it does not redo OOF selection. Reject --smoke with --final. Keep stage logs and the established result schemas; smoke pseudo files use dates ≤Aug 31 and f4 thresholds, never September values.

At OOF stage select (τ,h) separately for each fold and pass that fold's pair to BB/B1/B2/B3/B4 and ablations. At calibration stage create inner predictions and p* for every (model,variant,fold); after H/IO/KAN additions, recalibrate, rebuild p* and metrics before selecting. Compute outer NLL only in metrics, from a fixed fit state and outer observations. Record K4/isotonic run/not_run flags and calibration counts. selection.json records submitted_model=B4, submitted_variant, submitted ID, selected flags/parameters, final p* and statuses. Pass the exact selected (model,variant) through final ref calibration, p*, error analysis, FN/FP and submission. Final comparator B0/B0p/B1 stays separate.

Acceptance: smoke subprocess returns 1,344 pseudo-test rows dated Aug 18–31, no real-unseal call and correct selected variant ID. Synthetic H selection verifies same H OOF calibrator, p*, analysis and submission ID. Default/--no-final emits no real-test predictions. No K4/iso run leaves explicit not_run records with no fabricated score.

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
- [ ] **Step 5: Full suite and verify**

---

### Task 20: Development run and exploratory gate

Run the default runner (or --no-final) with the sealed loader; inspect only pre-September OOF, risk, 53-day bootstrap and selection.json. Compare B4 variants to fixed 100-round B1. If the gate fails, run the registered H→IO→KAN loop only until success or 2026-10-03, reporting attempted count and exploratory-CI caveat. Keep September sealed. Acceptance: all outputs end by Aug 31; selection is a B4 variant; missing p* keys or empty calibration status are visible, not silently filled.

---

### Task 21: Improvement loop ① — B4-H hybrid centre

Files: gmst/baselines.py, gmst/hmm.py, gmst/run_all.py and focused tests.

Implement b1_centre(panel,train_idx,tau,half_life,C,protocol="A+",rounds=100,n_blocks=5) using the existing 19-quantile B1 slot model and its sorted τ=0.5 prediction. On issued days, fit B1 only on that fold's permitted train dates. On HMM training days, use five chronological cross-fit blocks, training each block centre on the other train blocks so its own target labels are absent. This is internal training construction; issued and calibration-day B1 centres remain strictly forward-only. Reuse B1's quantile fit/predict path rather than training a duplicate B1 stack. The variant H uses model B4, variant H and submitted ID B4-H. It has its own full, tune and cal fits, same-variant calibrator, p* and exploratory gate comparison.

Acceptance: issued H centre equals fold-safe B1 median, training centres never use their own labels; issued/calibration centres never use future days, selected H survives through analysis/final/submission. Keep B1 benchmark default at 100 rounds.

---

### Task 22: Improvement loop ② — B4-IO conditional decoder

Files: gmst/hmm.py, gmst/run_all.py and focused tests.

Keep the 13-column u and 3-column v, ordered state means, time-varying σ/φ and the existing IO penalty. select_io_lambda(panel,flags,fold,grid=(0,0.01,0.1,1),...) chooses λ for exactly one fold on its tuning_idx; no pooled f1–f4 scoring. Use fixed λ=0.01 with status=default_empty if that fold has no usable tuning target. The final test fit chooses λ anew from its pre-September train. The runner stores fold,lambda_io,nll_tune,n_tune,status; variant IO receives fold-specific λ and produces calibration inner predictions and p* before gate comparison.

Acceptance: changing other folds' labels or this fold's outer labels cannot change its λ or first issued prediction. The IO model and variant survive to the result tables if selected.

---

### Task 23: Improvement loop ③ — B4-KAN periodic-spline transition

Files: gmst/hmm.py, gmst/run_all.py and focused tests.

Keep cubic cyclic B-splines with M∈{8,12,16}. z_features(kan=True,M) returns 2M+3 columns: two operating-state spline blocks followed by sat,sun,hol; no separate op column and no transition intercept. For K=3,M=12 this is 162 transition parameters. spline_penalty sums cyclic first differences inside each M block. select_kan(panel,flags,fold,...) chooses M and penalty λ on that fold's tuning_idx only; empty tuning uses (M=12,λ=0.01) with status=default_empty. The final pre-September train chooses its own pair. Record fold,M,lambda_spl,nll_tune,n_tune,status; add KAN inner predictions/p* before gate comparison.

Acceptance: basis rows sum to one, KAN feature count is 2M+3, no redundant op parameter, fold j outer-label perturbation leaves its choice and first issued prediction unchanged.

---

### Task 24: Single real sealed final run and verification

Only after the B4 family/variant, settings and thresholds are fixed in selection.json, invoke python -m gmst.run_all --final once. This is the only real unseal. It evaluates the chosen B4 variant plus B0/B0p/B1 comparison; same-(model,variant) pre-September OOF fits the final calibrator, and final train cal days determine final p*. Save 1,344 September submission rows and evaluation mask. Never reselect from test metrics. If an execution defect forces a rerun, document the bug and exact rerun reason.

Before --final, run default/full development checks, smoke pseudo-test twice on Aug 18–31 for same-device ≤1e-6 numerical reproducibility, and synthetic loader-unseal/selected-H integration tests. After --final, validate schema, shape, finite issued fields and chosen model ID without retraining or opening the sealed data again. No second full runner execution for determinism. Final results tests are gated on test_predictions.csv existence, not merely selection.json.

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
- [ ] **Step 5: Full suite and verify**

---

### Task 26: README, PONYTAIL-DEBT.md ledger, final verification, package

**Files:**
- Modify: `README.md` (currently empty), `tests/test_style.py` (append)
- Create: `PONYTAIL-DEBT.md` (at `WT` root, i.e. the repository root after merge)

**Contract:**
- `README.md` (Korean, FR-101, no affiliation/logo, FR-7), sections in order: 개요 (문제정의 H=96 / L=672 / 00:00 발행 / C = fold 학습 가동일 일최대 q50·q75·q90, 테스트 186.5·198·210.7); 환경 (`uv sync` 또는 `pip install -r requirements.txt`, torch cu126 index, `CUDA_VISIBLE_DEVICES=""`로 CPU 실행); 단일 명령 (`uv run python -m gmst.run_all`, `--no-final`, `--smoke`, `--final`, `--occ-floor`, `--package`); 산출물 표 (file → 보고서 장, PRD §6 table); 데이터 주의점 요약 (DC1–DC16 처리 규칙 한 줄씩); 모델 선택 (B1은 벤치마크, 제출은 항상 B4 계열; 게이트 = B1 대비 ΔMAE·Δ평균Brier CI 상한 < 0; 개선 루프 B4-H → B4-IO → B4-KAN과 `selection.json`의 시도 목록·개수·다중비교 주의문, B1 대비 잔여 격차); 제출파일 스키마 (`test_predictions.csv` 33열, `model` 열 의미 = `select_variant`가 정한 B4 변형 ID `B4`/`B4-H`/`B4-IO`/`B4-KAN`, `eval_mask.csv`); 외부자료 (`HOLIDAYS_2021` 하드코딩과 출처 한국천문연구원 특일정보, 한전 산업용(을) 2021 요금표와 출처, 외부 기상 미사용과 근거 = oracle-weather `AWstar − main` 결과(휴리스틱 상한, 수치는 `results/bootstrap.csv`에서 인용)); ₩ 가정 (전력 단위 "15분 평균 kW" 가정, 상대변화(%) 병기, 래칫 바닥 222 → 7–8월 기본요금 절감 ≈ 0, kWh당 가산요금은 `d_kwh ≈ 0`일 때만 상쇄); 재현성 (seed 0, 날짜별 MC seed, CUDA/CPU, 봉인 전 pseudo-test 같은 장치 재실행 차 ≤ 1e-6); 폴더 구조; 제출 zip (`dist/kamp_power_src.zip`).
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
- [ ] **Step 5: Verify**

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

## Current review resolution

The approved 2026-09-23 review R1–R8 and modeling notes are resolved in the shared contracts and tasks above. Historical code snippets in unaffected tasks remain examples; where any old example conflicts, use the reviewed contract, PRD and v2 as amended. The 26 task headings and requirement mapping remain intact.

- Data: derive 53 usable OOF days from finite targets; point/day risk denominators differ.
- Time order: split each outer train into fit, tuning and calibration days; choose τ/h, IO/KAN and early stopping per fold, never by pooling later folds. Final September settings are chosen from pre-September train only.
- Risk: every model and variant supplies cal-day predictions and a p* key; calibrators use train-period rows for outer OOF and same-variant pre-September OOF for final.
- Run state: K4 and isotonic are explicit run/not_run options. Omitted ablations have no manufactured scores.
- Evaluation: outer NLL is computed after fit. Missing-predecessor AR likelihood is a stationary-reset approximation. KAN uses 2M+3 features and 162 transition parameters at K=3,M=12.
- Submission: smoke and reproducibility use Aug 18–31 pseudo-test; only explicit --final unseals September after selection. Error analysis and submission carry model plus variant.
- Interpretation: A+ assumes the observed operating flag stands in for a known 00:00 schedule. The post-selection CI gate is exploratory; B1 remains a fixed 100-round benchmark and never the submitted model.

## Submission handoff checklist

| Deliverable | Owner role | Target date | Verification |
|---|---|---|---|
| Six-chapter report PDF and figures | Report owner | 2026-10-06 | Sections 1–6 use frozen results and disclose all tried variants, CI caveat, gaps, schedule-proxy assumption |
| Survey completion capture | Submission owner | 2026-10-06 | Capture opens and identifying details are removed where required |
| Presentation PDF and PPT | Presentation owner | 2026-10-06 | Both formats open and use final selected B4 ID |
| Source ZIP, requirements, README, predictions and mask | Code owner | 2026-10-06 | Package check, 1,344-row prediction schema, no credentials or affiliation |
| Blind review and portal submission receipt | Submission owner | 2026-10-08 23:59 | Files checked for affiliation/logo and portal confirms receipt |

2026-10-07–08 remain buffer days; the final portal deadline is 2026-10-08 23:59. Roles are assignments to be filled by the team, not invented people.
