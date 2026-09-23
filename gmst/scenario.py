"""Historical tariff and model-based schedule what-ifs, conditional on 15-min kW.

Rates exclude tax and uniform kWh adders; production conservation does not imply
energy conservation. Schedules assume operational feasibility, not causal effects.
"""
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING, Final, Literal, TypedDict, assert_never

import numpy as np
import polars as pl

from gmst.contracts import BoolArray, FloatArray, Panel, Prediction

if TYPE_CHECKING:
    from gmst.hmm import HMMState

type Season = Literal["summer", "springfall", "winter"]
type Option = Literal["I", "II", "III"]
type Transform = Literal["baseline", "shift_peak", "stagger_start", "avoid_high", "ease_peak"]


class Tariff(TypedDict):
    base: float
    summer: tuple[float, float, float]
    springfall: tuple[float, float, float]
    winter: tuple[float, float, float]


TARIFF: Final[dict[Option, Tariff]] = {
    "I": {"base": 7220, "summer": (61.6, 114.5, 196.6), "springfall": (61.6, 84.1, 114.8), "winter": (68.6, 114.7, 172.2)},
    "II": {"base": 8320, "summer": (56.1, 109.0, 191.1), "springfall": (56.1, 78.6, 109.3), "winter": (63.1, 109.2, 166.7)},
    "III": {"base": 9810, "summer": (55.2, 108.4, 178.7), "springfall": (55.2, 77.3, 101.0), "winter": (62.5, 108.6, 155.5)},
}
RATCHET_MONTHS: Final = (12, 1, 2, 7, 8, 9)
TRANSFORMS: Final[tuple[Transform, ...]] = ("shift_peak", "stagger_start", "avoid_high", "ease_peak")
RATES: Final = (0.1, 0.2, 0.3)


def season_name(month: int) -> Season:
    if not 1 <= month <= 12:
        raise ValueError(f"Month outside 1..12: {month}")
    if 6 <= month <= 8:
        return "summer"
    return "winter" if month <= 2 or month >= 11 else "springfall"


def tariff_holidays(tariff_0816: bool = True) -> set[date]:
    """Treat the 2021-08-16 substitute holiday as offpeak by assumption."""
    from gmst.splits import HOLIDAYS_2021

    return set(HOLIDAYS_2021) - (set() if tariff_0816 else {date(2021, 8, 16)})


def band(ts: datetime, holiday: bool, for_energy: bool) -> int:
    if holiday or ts.weekday() == 6 or ts.hour < 9 or ts.hour >= 23:
        return 0
    winter = ts.month <= 2 or ts.month >= 11
    peak = 10 <= ts.hour < 12 or (17 <= ts.hour < 20 or ts.hour == 22 if winter else 13 <= ts.hour < 17)
    return 2 if peak and not (for_energy and ts.weekday() == 5) else 1


def rate(ts: datetime, holiday: bool, option: Option = "II") -> float:
    return TARIFF[option][season_name(ts.month)][band(ts, holiday, True)]


def energy_won(y: FloatArray, ts: Sequence[datetime], holidays: set[date], option: Option = "II") -> float:
    """Sum finite quarter-hour average kW using the historical energy bands."""
    return float(sum(0.25 * value * rate(t, t.date() in holidays, option)
                     for value, t in zip(y, ts, strict=True) if np.isfinite(value)))


def billing_demand(y: FloatArray, ts: Sequence[datetime], holidays: set[date]) -> float:
    return max((float(value) for value, t in zip(y, ts, strict=True)
                if np.isfinite(value) and band(t, t.date() in holidays, False) >= 1), default=0.0)


def _timestamps(day: date) -> list[datetime]:
    return [datetime.combine(day, time()) + timedelta(minutes=15 * q) for q in range(96)]


def ratchet_floor(panel: Panel, month: int, exclude_dates: set[date], holidays: set[date]) -> tuple[float, datetime]:
    """Return observed 2021 demand floor; zero/month-start means no eligible reading."""
    # ponytail: 2020-12 자료 없음·계약전력 30% 하한 미적용·마스킹 점 제외로 래칫 바닥 계산, 실제 청구 이력이 오면 교체 (FR-85)
    season_name(month)
    best, at = 0.0, datetime(2021, month, 1)
    months = {m for m in RATCHET_MONTHS if m <= month} | {month}
    for d, day in enumerate(panel["dates"]):
        if day.year != 2021 or day.month not in months or day in exclude_dates or day in holidays or day.weekday() == 6:
            continue
        for q, ts in enumerate(_timestamps(day)):
            value = float(panel["Y"][d, q])
            if np.isfinite(value) and value > best and band(ts, False, False) >= 1:
                best, at = value, ts
    return best, at


def expected_p_app(month_max: FloatArray, floor: float) -> float:
    if month_max.size == 0 or np.isnan(month_max).any() or not np.isfinite(floor):
        raise ValueError(f"Invalid demand samples or floor: {floor}")
    return float(np.maximum(month_max, floor).mean())


def basic_won(p_app: float, option: Option = "II") -> float:
    return TARIFF[option]["base"] * p_app


def _schedule(P: FloatArray, s: float) -> FloatArray:
    if P.shape != (24,) or not np.isfinite(P).all() or (P < 0).any() or not 0 <= s <= 1:
        raise ValueError(f"Expected finite nonnegative 24-hour schedule and share in [0,1], got shape={P.shape}, share={s}")
    return P.copy()


def _redistribute(P: FloatArray, s: float, donors: BoolArray) -> FloatArray:
    out = _schedule(P, s)
    donors = donors & (P > 0)
    receivers = ~donors & (P > 0)
    if donors.any() and receivers.any():
        moved = s * P[donors]
        out[donors] -= moved
        out[receivers] += moved.sum() * P[receivers] / P[receivers].sum()
    return out


def shift_peak(P: FloatArray, day: date, s: float, holidays: set[date]) -> FloatArray:
    donors = np.array([band(datetime.combine(day, time(h)), day in holidays, False) == 2 for h in range(24)])
    return _redistribute(P, s, donors)


def stagger_start(P: FloatArray, s: float) -> FloatArray:
    out = _schedule(P, s)
    starts = (P > 0) & np.r_[True, P[:-1] <= 0]
    for h in np.flatnonzero(starts):
        if h < 23 and P[h + 1] > 0:
            out[h] -= s * P[h]
            out[h + 1] += s * P[h]
    return out


def avoid_high(P: FloatArray, s: float, y_med: FloatArray, c90: float) -> FloatArray:
    if y_med.shape != (96,) or not np.isfinite(y_med).all() or not np.isfinite(c90):
        raise ValueError(f"Expected finite 96-slot median and threshold, got shape={y_med.shape}, C90={c90}")
    return _redistribute(P, s, (y_med.reshape(24, 4) > c90).any(axis=1))


def ease_peak(P: FloatArray, s: float, peak_slot: int) -> FloatArray:
    if not 0 <= peak_slot < 96:
        raise ValueError(f"Peak slot outside 0..95: {peak_slot}")
    donors = np.zeros(24, dtype=bool)
    donors[np.arange(max(0, peak_slot - 2), min(96, peak_slot + 3)) // 4] = True
    return _redistribute(P, s, donors)


def c_change(P0: FloatArray, Pa: FloatArray) -> float:
    return float(np.abs(Pa - P0).sum() / (2 * P0.sum())) if P0.sum() > 0 else 0.0


SCN_DAY_COLS: Final = ["fold", "date", "scenario", "rate", "E_M", "M_median", "risk_C50", "risk_C75", "risk_C90", "risk_ok", "peak_time_mode", "C_change", "d_energy_won", "eta_star"]
SCN_MONTH_COLS: Final = ["month", "scenario", "rate", "n_days", "P_floor", "E_P_app_base", "E_P_app_scn", "d_demand_won", "d_energy_won", "d_kwh", "d_total_won", "d_total_pct", "C_change", "eta_star", "risk_ok_all"]
_DAY_SCHEMA: Final = dict(zip(SCN_DAY_COLS, [pl.String] * 3 + [pl.Float64] * 6 + [pl.Boolean, pl.Int64] + [pl.Float64] * 3, strict=True))
_MONTH_SCHEMA: Final = dict(zip(SCN_MONTH_COLS, [pl.Int64, pl.String, pl.Float64, pl.Int64] + [pl.Float64] * 10 + [pl.Boolean], strict=True))
type _Cell = str | float | int | bool


class _DayStats(TypedDict):
    day: date
    maxima: FloatArray
    energy: float
    kwh: float
    production: float
    movement: float
    risk_ok: bool


def _transform(name: Transform, P: FloatArray, day: date, share: float,
               base: Prediction, C: FloatArray, holidays: set[date]) -> FloatArray:
    match name:
        case "baseline":
            return _schedule(P, share)
        case "shift_peak":
            return shift_peak(P, day, share, holidays)
        case "stagger_start":
            return stagger_start(P, share)
        case "avoid_high":
            return avoid_high(P, share, base["y_median"], float(C[2]))
        case "ease_peak":
            return ease_peak(P, share, base["peak_time_mode"])
        case unreachable:
            assert_never(unreachable)


def _monthly(panel: Panel, groups: Mapping[tuple[int, Transform, float], list[_DayStats]],
             holidays: set[date]) -> pl.DataFrame:
    """Pair independent dates by path index, then apply the monthly demand floor."""
    rows: list[tuple[_Cell, ...]] = []
    for (month, name, share), days in groups.items():
        baseline = groups[(month, "baseline", 0.0)]
        floor, _ = ratchet_floor(panel, month, {d["day"] for d in baseline}, holidays)
        # ponytail: 날짜 간 경로 독립 짝짓기, 날짜 간 상관이 크면 일 랜덤효과 경로로 교체 (FR-85)
        app0 = expected_p_app(np.maximum.reduce([d["maxima"] for d in baseline]), floor)
        app = expected_p_app(np.maximum.reduce([d["maxima"] for d in days]), floor)
        demand = basic_won(app - app0)
        energy0 = sum(d["energy"] for d in baseline)
        delta_energy = sum(d["energy"] for d in days) - energy0
        delta_kwh = sum(d["kwh"] for d in days) - sum(d["kwh"] for d in baseline)
        total, denominator = demand + delta_energy, basic_won(app0) + energy0
        production = sum(d["production"] for d in days)
        change = sum(d["movement"] for d in days) / (2 * production) if production else 0.0
        rows.append((month, name, share, len(days), floor, app0, app, demand, delta_energy, delta_kwh,
                     total, 100 * total / denominator if denominator else float("nan"), change,
                     -total / change if change else float("nan"), all(d["risk_ok"] for d in days)))
    return pl.DataFrame(rows, schema=_MONTH_SCHEMA, orient="row")


def run_scenarios(panel: Panel, states: Mapping[str, "HMMState"], N: int = 2000,
                  holidays: set[date] | None = None, max_days: int | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Simulate f2–f4 operating days with protocol B and shared date-seeded draws.

    These are conditional model what-ifs. Unknown capacity limits, taxes, adders,
    missing December 2020 demand and the contract-demand floor remain unpriced.
    """
    from gmst.features import role_idx, usable_peak
    from gmst.hmm import forecast

    if N < 1 or (max_days is not None and max_days < 0) or set(states) - {"f2", "f3", "f4"}:
        raise ValueError(f"Expected f2–f4 states, N>0 and max_days>=0; got folds={tuple(states)}, N={N}, max_days={max_days}")
    holidays = tariff_holidays() if holidays is None else holidays
    usable = usable_peak(panel)
    rows: list[tuple[_Cell, ...]] = []
    groups: dict[tuple[int, Transform, float], list[_DayStats]] = {}
    scenarios: list[tuple[Transform, float]] = [("baseline", 0.0), *((t, s) for t in TRANSFORMS for s in RATES)]
    for fold, state in sorted(states.items()):
        training = [panel["dates"][int(i)] for i in state["train_idx"]]
        if state["protocol"] != "B" or any(d < date(2021, 7, 1) or d > date(2021, 8, 31) for d in training):
            raise ValueError(f"Scenario state must use July–August protocol B training: {fold}")
        targets = [int(d) for d in role_idx(panel, fold, "val") if usable[d] and panel["op"][d] == 1]
        if max_days is not None:
            targets = targets[:max_days]
        for d in targets:
            day = panel["dates"][d]
            if not date(2021, 7, 21) <= day <= date(2021, 8, 31) or any(t >= day for t in training):
                raise ValueError(f"Scenario target outside forward pre-September window: {day}")
            P0, ts = panel["X"]["생산량"][d, ::4], _timestamps(day)
            _schedule(P0, 0.0)
            base = forecast(state["model"], panel, d, state["m"], state["C"], "B", N, device=state["device"])
            energy0 = energy_won(base["y_mean"], ts, holidays)
            demand_slots = np.array([band(t, day in holidays, False) >= 1 for t in ts])
            for name, share in scenarios:
                Pa = _transform(name, P0, day, share, base, state["C"], holidays)
                pred = base
                if not np.array_equal(Pa, P0):
                    changed: Panel = {**panel, "X": {**panel["X"], "생산량": panel["X"]["생산량"].copy()}}
                    changed["X"]["생산량"][d] = np.repeat(Pa, 4)
                    pred = forecast(state["model"], changed, d, state["m"], state["C"], "B", N, device=state["device"])
                paths = pred["paths"]
                if paths is None or paths.shape != (N, 96) or not np.isfinite(paths).all():
                    raise ValueError(f"Scenario needs {N} finite 96-slot paths: {day}, {name}")
                energy = energy_won(pred["y_mean"], ts, holidays)
                delta, change = energy - energy0, c_change(P0, Pa)
                risk_ok = bool(pred["risk_raw"][2] <= base["risk_raw"][2])
                rows.append((fold, day.strftime("%Y.%m.%d"), name, share, pred["M_hat_mean"], pred["M_hat_median"],
                             *map(float, pred["risk_raw"]), risk_ok, pred["peak_time_mode"], change,
                             delta, -delta / change if change else float("nan")))
                groups.setdefault((day.month, name, share), []).append({"day": day,
                    "maxima": paths[:, demand_slots].max(axis=1) if demand_slots.any() else np.full(N, -np.inf),
                    "energy": energy, "kwh": float(0.25 * pred["y_mean"].sum()), "production": float(P0.sum()),
                    "movement": float(np.abs(Pa - P0).sum()), "risk_ok": risk_ok})
    return pl.DataFrame(rows, schema=_DAY_SCHEMA, orient="row"), _monthly(panel, groups, holidays)
