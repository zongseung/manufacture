import json
from pathlib import Path

import numpy as np
import polars as pl

from gmst.contracts import FloatArray


def summarize(frame: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    return frame.group_by(keys).agg(
        pl.len().alias("n_days"), pl.col("M_true").mean().alias("true_peak_mean"),
        pl.col("M_hat_median").mean().alias("predicted_peak_mean"),
        (pl.col("M_hat_median") - pl.col("M_true")).mean().alias("bias"),
        (pl.col("M_hat_median") - pl.col("M_true")).abs().mean().alias("peak_mae"),
        ((pl.col("peak_time_mode") - pl.col("peak_slot_true")).abs() <= 2).mean().alias("peak_hit2"),
    ).sort(keys)


def peak_interval_summary(frame: pl.DataFrame) -> pl.DataFrame:
    usable = frame.filter(pl.col("usable_peak"))
    aggregates = [pl.len().alias("n_days")]
    for level, low, high in ((50, "q25", "q75"), (80, "q10", "q90"), (90, "q05", "q95")):
        aggregates.extend([
            ((pl.col("M_true") >= pl.col(low)) & (pl.col("M_true") <= pl.col(high)))
            .mean().alias(f"coverage{level}"),
            (pl.col(high) - pl.col(low)).mean().alias(f"width{level}_kw"),
        ])
    strata = (("all", pl.lit(True)), ("nonop", pl.col("op") == 0),
              ("op", pl.col("op") == 1), ("high", pl.col("M_true") > pl.col("C90")))
    return pl.concat([usable.filter(predicate).group_by("model").agg(aggregates)
                      .with_columns(pl.lit(name).alias("stratum")) for name, predicate in strata])


def basic_split_rhat(chains: FloatArray) -> FloatArray:
    """Classical split-chain variance ratio; no rank normalization or folding."""
    half = chains.shape[1] // 2
    if half < 2:
        return np.full(chains.shape[2], np.nan)
    split = np.concatenate((chains[:, :half], chains[:, -half:]), axis=0)
    within = np.var(split, axis=1, ddof=1).mean(axis=0)
    between = half * np.var(split.mean(axis=1), axis=0, ddof=1)
    variance = (half - 1) / half * within + between / half
    ratio = np.divide(variance, within, out=np.full_like(within, np.inf), where=within > 0)
    ratio[np.ptp(chains, axis=(0, 1)) == 0] = np.nan
    return np.sqrt(ratio)


def convergence_report(out: Path, folds: tuple[str, ...], seeds: tuple[int, ...],
                       models: list[str]) -> pl.DataFrame:
    rows = []
    for fold in folds:
        for model in models:
            for stage in ("outer", "inner"):
                paths = [out / "jobs" / f"seed{seed}_{fold}_{model}" for seed in seeds]
                if stage == "inner":
                    paths = [path / "inner_posterior" for path in paths]
                traces = [pl.read_csv(path / "posterior_trace.csv").drop("draw") for path in paths]
                assert all(trace.columns == traces[0].columns for trace in traces)
                chains = np.stack([trace.to_numpy() for trace in traces])
                rhats = basic_split_rhat(chains)
                for index, parameter in enumerate(traces[0].columns):
                    if np.ptp(chains[:, :, index]) == 0:
                        continue
                    value = float(rhats[index])
                    enough = len(seeds) >= 2 and chains.shape[1] >= 4
                    rows.append({"fold": fold, "model": model, "stage": stage, "parameter": parameter,
                                 "n_seeds": len(seeds), "draws_per_seed": chains.shape[1],
                                 "basic_split_rhat": value, "sufficient_chains": enough,
                                 "flagged": not enough or not np.isfinite(value) or value > 1.05})
    detail = pl.DataFrame(rows)
    detail.write_csv(out / "split_rhat.csv")
    summary = detail.group_by("model", "fold", "stage").agg(
        pl.len().alias("n_parameters"), pl.col("basic_split_rhat").max().alias("max_basic_split_rhat"),
        pl.col("flagged").sum().alias("n_flagged"), (~pl.col("flagged").any()).alias("passed"),
    ).sort("model", "fold", "stage")
    summary.write_csv(out / "convergence_summary.csv")
    (out / "convergence_method.json").write_text(json.dumps({
        "method": "classical split R-hat, not rank-normalized or folded",
        "flag_threshold": 1.05, "constant_parameters": "excluded",
        "limitations": "Basic R-hat only; passing is not proof of convergence or adequate tail ESS.",
        "seeds": seeds,
    }, indent=2) + "\n")
    return summary
