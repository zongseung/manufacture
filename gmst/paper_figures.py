"""Vector paper figures from saved development predictions; no model fitting."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.figure import Figure

from gmst import ROOT, bat

MODELS = ("BAT", "BAT_no_transfer", "BAT_pooled_gaussian", "BAT_conditional_gaussian", "B0_kind", "M2_tuned")
LABELS = ("BAT", "No transfer", "Pooled", "Conditional", "B0_kind", "LightGBM")
COLORS = ("#707070", "#aaaaaa", "#cc7733", "#006c91", "#278153", "#925b99")
CONDITIONAL = MODELS[3]
STYLE = {"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
         "ps.fonttype": 42, "axes.grid": True, "grid.alpha": .18, "grid.linewidth": .5}


def select_examples(slots: pl.DataFrame, days: pl.DataFrame) -> pl.DataFrame:
    errors = (slots.filter((pl.col("model") == CONDITIONAL) & pl.col("y_true").is_finite()
                           & pl.col("y_median").is_finite()).group_by("date")
              .agg((pl.col("y_true") - pl.col("y_median")).abs().mean().alias("daily_mae")))
    errors = errors.join(days.filter(pl.col("usable_peak")), on="date", validate="1:1")
    strata = (("Nonop.", pl.col("op") == 0),
              ("Op. ≤ C90", (pl.col("op") == 1) & (pl.col("M_true") <= pl.col("C90"))),
              ("High peak", (pl.col("op") == 1) & (pl.col("M_true") > pl.col("C90"))))
    selected = []
    for label, predicate in strata:
        group = errors.filter(predicate)
        assert group.height, f"No usable dates in {label}"
        selected.append(group.with_columns((pl.col("daily_mae") - pl.col("daily_mae").median()).abs()
                                            .alias("distance"))
                        .sort("distance", "date").head(1).with_columns(pl.lit(label).alias("stratum")))
    return pl.concat(selected)


def forecast_figure(slots: pl.DataFrame, days: pl.DataFrame) -> Figure:
    examples = select_examples(slots, days)
    fig, axes = plt.subplots(3, 1, figsize=(7.1, 5.5), sharex=True, layout="constrained")
    for ax, row in zip(axes, examples.iter_rows(named=True), strict=True):
        rows = slots.filter(pl.col("date") == row["date"]).sort("datetime")
        conditional = rows.filter(pl.col("model") == CONDITIONAL)
        x = np.arange(conditional.height) / 4
        ax.fill_between(x, conditional["q05"], conditional["q95"], color=COLORS[3], alpha=.15, label="90% PI")
        ax.fill_between(x, conditional["q25"], conditional["q75"], color=COLORS[3], alpha=.25, label="50% PI")
        for m, style in ((0, "--"), (4, ":"), (5, "-."), (3, "-")):
            curve = rows.filter(pl.col("model") == MODELS[m])
            ax.plot(x, curve["y_median"], style, color=COLORS[m], lw=1.1, label=LABELS[m])
        ax.plot(x, conditional["y_true"], color="black", lw=1.05, label="Observed")
        ax.set(title=f"{row['stratum']} · {row['date'][5:]} · MAE {row['daily_mae']:.1f}", ylabel="kW", xlim=(0, 24))
    axes[-1].set(xticks=np.arange(0, 25, 4), xlabel="Hour")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=7)
    return fig


def comparison_figure(metrics: pl.DataFrame) -> Figure:
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.8), sharey=True, layout="constrained")
    for ax, (metric, title) in zip(axes, (("r2", "Slot $R^2$"), ("rmse", "Slot RMSE (kW)"),
                                        ("peak_mae", "Peak MAE (kW)")), strict=True):
        for index, model in enumerate(MODELS):
            row = metrics.filter((pl.col("stratum") == "all") & (pl.col("metric") == metric) & (pl.col("model") == model)).row(0, named=True)
            ax.plot([row["seed_min"], row["seed_max"]], [index, index], color=COLORS[index], lw=2)
            ax.scatter(row["value"], index, color=COLORS[index], s=28, zorder=3)
            ax.annotate(f"{row['value']:.4f}" if metric == "r2" else f"{row['value']:.2f}",
                        (row["value"], index), xytext=(4, 5), textcoords="offset points", fontsize=7)
        ax.set(title=title, yticks=np.arange(6), yticklabels=LABELS, ylim=(5.6, -.65))
        ax.margins(x=.3)
    return fig


def peak_figure(metrics: pl.DataFrame) -> Figure:
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 2.8), sharey=True, sharex=True, layout="constrained")
    for ax, (stratum, title) in zip(axes, (("all", "All (51)"), ("nonop", "Nonop. (14)"),
                                         ("op", "Op. (37)"), ("high", "High (13)")), strict=True):
        for index, model in enumerate(MODELS):
            row = metrics.filter((pl.col("stratum") == stratum) & (pl.col("metric") == "peak_mae") & (pl.col("model") == model)).row(0, named=True)
            ax.plot([row["seed_min"], row["seed_max"]], [index, index], color=COLORS[index], lw=2)
            ax.scatter(row["value"], index, color=COLORS[index], s=25, zorder=3)
        ax.set(title=title, xlabel="kW", yticks=np.arange(6), yticklabels=LABELS, ylim=(5.5, -.5), xlim=(-1, 33), xticks=[0, 10, 20, 30])
    return fig


def coverage_figure(intervals: pl.DataFrame) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.7), sharey=True, layout="constrained")
    labels = []
    for index, row in enumerate(intervals.iter_rows(named=True)):
        color = COLORS[MODELS.index(row["model"])]
        labels.append(f"{LABELS[MODELS.index(row['model'])]}: {row['target']}")
        for ax, metric in zip(axes, ("coverage", "width"), strict=True):
            value = row[metric] * (100 if metric == "coverage" else 1)
            ax.scatter(value, index, color=color, s=30, zorder=3)
            ax.annotate(f"{value:.1f}", (value, index), xytext=(5, 4), textcoords="offset points", fontsize=7)
    axes[0].axvline(90, color="black", ls="--", lw=.8, label="Nominal")
    axes[0].set(xlabel="Coverage (%)", xlim=(60, 104))
    axes[0].legend(loc="lower left", fontsize=7)
    axes[1].set(xlabel="Width (kW)", xlim=(0, intervals["width"].max() * 1.25))
    axes[0].set(yticks=np.arange(len(labels)), yticklabels=labels, ylim=(len(labels) - .5, -.7))
    return fig


def attention_figure(traces: list[pl.DataFrame]) -> Figure:
    matrices = [np.mean([bat.attention(theta) for theta in trace.select([f"theta[2,{j}]" for j in range(5)]).to_numpy()], axis=0) for trace in traces]
    fig, axes = plt.subplots(1, len(traces), figsize=(7.1, 2.55), sharey=True, layout="constrained")
    for seed, (ax, matrix) in enumerate(zip(axes, matrices, strict=True)):
        mesh = ax.pcolormesh(np.arange(25), np.arange(97) / 4, matrix, cmap="cividis", vmin=0, vmax=max(float(m.max()) for m in matrices), rasterized=False, edgecolors="face", linewidth=.05)
        ax.set(title=f"Seed {seed}", xlabel="Prod. hour", xticks=[0, 8, 16, 24], yticks=[0, 8, 16, 24], ylim=(24, 0))
        ax.grid(False)
    axes[0].set_ylabel("Forecast hour")
    colorbar = fig.colorbar(mesh, ax=axes, label="Attention", fraction=.025, pad=.03)
    colorbar.solids.set_rasterized(False)
    colorbar.solids.set_edgecolor("face")
    return fig


def noise_figure(traces: list[pl.DataFrame]) -> Figure:
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.6), layout="constrained")
    for ax, (parameter, label) in zip(axes, (("sigma_u_kw", "Day SD (kW)"), ("sigma_eta_kw", "Innov. SD (kW)"), ("rho", "AR ρ")), strict=True):
        for group, color in ((0, "#278153"), (1, "#006c91")):
            boxes = ax.boxplot([trace[f"{parameter}[{group}]"].to_numpy() for trace in traces], positions=np.arange(3) + (group - .5) * .3,
                              widths=.24, patch_artist=True, showfliers=False, manage_ticks=False,
                              medianprops={"color": "black", "linewidth": .8})
            for box in boxes["boxes"]:
                box.set(facecolor=color, alpha=.6)
            ax.plot([], [], color=color, lw=5, alpha=.6, label=("Nonop.", "Op.")[group])
        ax.set(xticks=np.arange(3), xticklabels=["S0", "S1", "S2"], title=label)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, loc="outside lower center", ncol=2)
    return fig


def generate_figures(source: Path = ROOT / "results_v3/conditional_collapsed", out: Path = ROOT / "paper/figs") -> list[Path]:
    """Regenerate PDFs from saved OOF predictions and record input hashes/selection."""
    inputs: dict[str, str] = {}

    def read(path: Path) -> pl.DataFrame:
        inputs[str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return pl.read_csv(path)

    slots = read(source / "seed0/oof_slots.csv")
    peak = read(source / "seed0/peak_intervals.csv")
    days = peak.filter(pl.col("model") == CONDITIONAL)
    assert set(slots["fold"]) == set(peak["fold"]) == {"f1", "f2", "f3", "f4"}
    assert slots["date"].max() < "2021.09.01"
    metrics = read(source.parent / "extended_metrics/metrics.csv").filter(pl.col("fold") == "pooled")
    intervals = []
    for seed in range(3):
        seed_slots = slots if seed == 0 else read(source / f"seed{seed}/oof_slots.csv")
        seed_peak = peak if seed == 0 else read(source / f"seed{seed}/peak_intervals.csv")
        for model in (MODELS[0], MODELS[2], MODELS[3]):
            for target, frame, truth in (("slot", seed_slots, "y_true"), ("peak", seed_peak, "M_true")):
                rows = frame.filter(pl.col("model") == model)
                if seed > 0 and model == MODELS[0] or rows.is_empty():
                    continue
                if target == "peak":
                    rows = rows.filter(pl.col("usable_peak"))
                rows = rows.filter(pl.all_horizontal(pl.col(c).is_finite() for c in (truth, "q05", "q95")))
                intervals.append({"model": model, "target": target, "seed": seed, "n": rows.height,
                                  "coverage": rows.select(((pl.col(truth) >= pl.col("q05")) & (pl.col(truth) <= pl.col("q95"))).mean()).item(),
                                  "width": rows.select((pl.col("q95") - pl.col("q05")).mean()).item()})
    interval_frame = pl.DataFrame(intervals).group_by("model", "target", maintain_order=True).agg(pl.col("coverage").mean(), pl.col("width").mean(), pl.col("n").min(), pl.col("seed").n_unique().alias("n_seeds"))
    traces = [read(source / f"jobs/seed{seed}_f4_{CONDITIONAL}/posterior_trace.csv") for seed in range(3)]
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    with plt.rc_context(STYLE):
        figures = {"forecasts": forecast_figure(slots, days), "model_comparison": comparison_figure(metrics),
                   "peak_strata": peak_figure(metrics), "coverage": coverage_figure(interval_frame),
                   "attention": attention_figure(traces), "noise": noise_figure(traces)}
        for name, fig in figures.items():
            path = out / f"conditional_{name}.pdf"
            fig.savefig(path, metadata={"Creator": "gmst.paper_figures", "CreationDate": None, "ModDate": None})
            plt.close(fig)
            paths.append(path)
    manifest = {"input_sha256": inputs, "code_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in (Path(__file__), ROOT / "gmst/bat.py")},
                "pdfs": [p.name for p in paths], "examples": select_examples(slots, days).select("stratum", "date", "fold", "daily_mae", "C90", "M_true").to_dicts(),
                "example_seed": 0, "example_selection": "Closest to stratum median daily slot MAE, finite pairs only; date breaks ties; usable peak dates only",
                "intervals": interval_frame.to_dicts(), "refitted": False, "september_opened": False,
                "seed_policy": "Mean of seed-specific metrics on identical days; baseline evaluated once; min/max bars are computational ranges, not confidence intervals",
                "posterior": "f4 outer only, kind 2 attention, seeds separate; basic split-Rhat failed overall; boxplots 25/50/75%, whiskers 1.5 IQR; outliers hidden",
                "high_peak": "Retrospective observed peak > training-fold C90; 13 days",
                "coverage": "Finite observed slots; usable daily peaks; original BAT peak quantiles unavailable"}
    (out / "conditional_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate vector paper PDFs from saved development predictions.")
    parser.add_argument("--source", type=Path, default=ROOT / "results_v3/conditional_collapsed")
    parser.add_argument("--out", type=Path, default=ROOT / "paper/figs")
    args = parser.parse_args()
    for path in generate_figures(args.source, args.out):
        print(path)


if __name__ == "__main__":
    main()
