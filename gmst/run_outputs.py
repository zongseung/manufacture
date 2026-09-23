"""Delivery schemas and serialization shared by development and final runs."""

from datetime import date, datetime, time, timedelta
from typing import Final

import numpy as np
import polars as pl

from gmst.contracts import FloatArray, Prediction

MODEL_ID: Final = {"main": "B4", "H": "B4-H", "IO": "B4-IO", "KAN": "B4-KAN"}
QCOLS: Final = [f"q{value:02d}" for value in range(5, 100, 5)]
PRED_COLS: Final = [
    "datetime",
    "track",
    "model",
    "y_mean",
    "y_median",
    *QCOLS,
    "M_hat_median",
    "M_hat_mean",
    "peak_time_mode",
    "risk_C50",
    "risk_C75",
    "risk_C90",
    "C50",
    "C75",
    "C90",
]
type Json = str | int | float | bool | None | Sequence[Json] | Mapping[str, Json]


def clean_json(value: Json) -> Json:
    """Use JSON null for undefined scientific statistics."""
    match value:
        case float():
            return value if np.isfinite(value) else None
        case list():
            return [clean_json(item) for item in value]
        case dict():
            return {key: clean_json(item) for key, item in value.items()}
        case _:
            return value


def submission_rows(
    day: date, prediction: Prediction, variant: str, thresholds: FloatArray, risk: FloatArray
) -> pl.DataFrame:
    """Format issued full-day values; evaluation masks never enter this path."""
    quantiles = prediction["q"]
    if quantiles is None or quantiles.shape != (19, 96):
        raise ValueError("submission requires 19 issued quantiles for all 96 slots")
    start = datetime.combine(day, time())
    peak = prediction["peak_time_mode"]
    return pl.DataFrame(
        {
            "datetime": [(start + timedelta(minutes=15 * q)).strftime("%Y.%m.%d %H:%M:%S") for q in range(96)],
            "track": ["MAIN"] * 96,
            "model": [MODEL_ID[variant]] * 96,
            "y_mean": prediction["y_mean"],
            "y_median": prediction["y_median"],
            **dict(zip(QCOLS, quantiles, strict=True)),
            "M_hat_median": [prediction["M_hat_median"]] * 96,
            "M_hat_mean": [prediction["M_hat_mean"]] * 96,
            "peak_time_mode": [f"{peak // 4:02d}:{15 * (peak % 4):02d}"] * 96,
            **{f"risk_C{c}": np.full(96, risk[j]) for j, c in enumerate((50, 75, 90))},
            **{f"C{c}": np.full(96, thresholds[j]) for j, c in enumerate((50, 75, 90))},
        }
    ).select(PRED_COLS)


from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import TypedDict

from gmst import evaluate as ev
from gmst.contracts import Panel
from gmst.hmm_forecast import HMMState


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Predeclared run budgets; smoke is a schema exercise with smaller fits."""

    smoke: bool = False
    occ_floor: bool = False

    @property
    def folds(self) -> tuple[str, ...]:
        return ("f4",) if self.smoke else ("f1", "f2", "f3", "f4")

    @property
    def epochs(self) -> int:
        return 2 if self.smoke else 300

    @property
    def paths(self) -> int:
        return 50 if self.smoke else 2000

    @property
    def rounds(self) -> int:
        return 10 if self.smoke else 100

    @property
    def bootstrap(self) -> int:
        return 20 if self.smoke else 2000


class RunData(TypedDict):
    panel: Panel
    tables: dict[str, pl.DataFrame]
    metadata: dict[str, Json]
    states: dict[tuple[str, str], HMMState]
    stars: ev.PStarTable
    variant: str
    components: list[str]


@contextmanager
def stage(number: int, name: str) -> Iterator[None]:
    """Emit one completion record per requested pipeline stage."""
    from gmst.hmm_core import DEVICE

    started = perf_counter()
    yield
    print(f"[{number:02d}] {name} {perf_counter() - started:.1f}s device={DEVICE.type}", flush=True)


def group(frame: pl.DataFrame, model: str, variant: str) -> pl.DataFrame:
    return frame.filter((pl.col("model") == model) & (pl.col("variant") == variant))


def refresh(tables: dict[str, pl.DataFrame], panel: Panel) -> ev.PStarTable:
    """Rebuild calibration and p* after every new candidate is appended."""
    days, inner = ev.calibrate_oof(tables["oof_days.csv"], "platt", tables["inner_days.csv"])
    if inner is None:
        raise ValueError("runner requires inner calibration rows")
    tables["oof_days.csv"], tables["inner_days.csv"] = days, inner
    keys = [(str(m), str(v), str(f)) for m, v, f in days.select(ev.KEYS).unique().iter_rows()]
    stars = ev.p_star_table(inner, keys)
    tables["calibration_status.csv"] = ev.p_star_status(inner, keys)
    tables["metrics.csv"] = pl.concat(
        [
            ev.point_metrics(tables["oof_slots.csv"], days, panel),
            ev.risk_metrics(ev.add_climatology(days, panel), panel, stars),
        ],
        how="diagonal_relaxed",
    )
    return stars


def candidate(tables: dict[str, pl.DataFrame], panel: Panel, variant: str, B: int) -> ev.Candidate:
    """Compare one B4 candidate to the fixed B1 benchmark."""
    rows = ev.compare(
        group(tables["oof_slots.csv"], "B4", variant),
        group(tables["oof_days.csv"], "B4", variant),
        group(tables["oof_slots.csv"], "B1", "main"),
        group(tables["oof_days.csv"], "B1", "main"),
        panel,
        f"{MODEL_ID[variant]}-B1",
        B=B,
    )
    boot = pl.DataFrame(rows)
    tables["bootstrap.csv"] = (
        pl.concat([tables["bootstrap.csv"], boot], how="diagonal_relaxed") if "bootstrap.csv" in tables else boot
    )
    metrics = group(tables["metrics.csv"], "B4", variant).filter(
        (pl.col("fold") == "pooled") & (pl.col("stratum") == "all")
    )
    mae = float(metrics.filter(pl.col("metric") == "mae")["value"].to_numpy()[0])
    brier = metrics.filter(pl.col("metric").is_in([f"brier_platt_{c}" for c in ev.CS]))["value"].to_numpy().mean()
    gap = {str(row["metric"]): [number(row[col]) for col in ("delta", "ci_lo", "ci_hi")] for row in rows}
    return {
        "mae": float(mae),
        "brier_mean": float(brier),
        "gate_met": ev.gate_pass(boot, f"{MODEL_ID[variant]}-B1"),
        "gap": gap,
    }


def pstar_json(stars: ev.PStarTable) -> dict[str, Json]:
    """Keep model/variant/fold identity in the serialized threshold table."""
    return {
        model: {
            variant: {fold: dict(values) for (m, v, fold), values in stars.items() if (m, v) == (model, variant)}
            for variant in sorted({v for m, v, _ in stars if m == model})
        }
        for model in sorted({m for m, _, _ in stars})
    }


@dataclass(frozen=True, slots=True)
class FrozenRun:
    variant: str
    tau: float
    half_life: int
    epochs: int
    components: tuple[str, ...]
    lambda_io: float
    M: int
    lambda_spl: float
    occ_floor: bool
    smoke: bool
    thresholds: tuple[float, float, float]


def number(value: Json) -> float:
    """Parse numeric artifact fields without accepting strings or containers."""
    match value:
        case bool():
            raise ValueError("boolean is not a numeric result field")
        case int() | float():
            return float(value)
        case _:
            raise ValueError("expected a numeric result field")


def parse_selection(value: Json, *, allow_smoke: bool = False) -> FrozenRun:
    """Reject incomplete or mismatched selection before the loader may unseal."""
    match value:
        case {
            "frozen": True,
            "mode": str(mode),
            "submitted_model": "B4",
            "submitted_variant": str(variant),
            "submitted": str(identifier),
            "tau": float(tau) | int(tau),
            "half_life": int(h),
            "final_epochs": int(epochs),
            "components": list(components),
            "final_settings": dict(settings),
            "thresholds": [c50, c75, c90],
        }:
            if variant not in MODEL_ID or identifier != MODEL_ID[variant]:
                raise ValueError("selection.json has an invalid B4 variant identity")
            if mode not in ("development", "smoke") or (mode == "smoke" and not allow_smoke):
                raise ValueError("real final requires a full development selection.json")
            if any(component not in ("H", "IO", "KAN") for component in components):
                raise ValueError("selection.json has unknown model components")
            if (variant == "main") != (len(components) == 0) or (variant != "main" and variant not in components):
                raise ValueError("selection.json identity and adopted components disagree")
            if not np.isfinite(tau) or tau <= 0 or h <= 0 or epochs <= 0:
                raise ValueError("selection.json has invalid frozen numeric settings")
            thresholds = number(c50), number(c75), number(c90)
            lambda_io = number(settings.get("lambda_io", 0.01))
            lambda_spl = number(settings.get("lambda_spl", 0.01))
            M = number(settings.get("M", 12))
            if (
                not np.isfinite([*thresholds, lambda_io, lambda_spl, M]).all()
                or not (0 < thresholds[0] <= thresholds[1] <= thresholds[2])
                or min(lambda_io, lambda_spl) < 0
                or M < 4
                or M != int(M)
            ):
                raise ValueError("selection.json has invalid thresholds or penalties")
            names = tuple(str(component) for component in components)
            return FrozenRun(
                variant,
                float(tau),
                h,
                epochs,
                names,
                lambda_io,
                int(M),
                lambda_spl,
                bool(settings.get("occ_floor", False)),
                mode == "smoke",
                thresholds,
            )
        case _:
            raise ValueError("selection.json is not a complete frozen B4 run")
