"""Compact vector figures for report/ from saved development outputs; no fitting, no September data."""
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.figure import Figure

from gmst import ROOT
from gmst.paper_figures import STYLE

CBAT = "BAT_conditional_gaussian_fixed_attention"
LABELS = ("C-BAT", "BAT", "B0_kind", "LightGBM")
COLORS = dict(zip(LABELS, ("#006c91", "#707070", "#278153", "#925b99"), strict=True))
SLOT_MODELS = {"C-BAT": CBAT, "BAT": "BAT", "LightGBM": "M2_tuned"}
WIDTH = 6.3
R = ROOT / "results_v3"


def table_label(label: str) -> str:
    return "LightGBM" if label.startswith("LightGBM") else label


def peak_hours_figure(days: pl.DataFrame) -> Figure:
    op = days.filter(pl.col("kind") > 0)
    hours = np.arange(24)
    high = [op.filter(pl.col("high") & (pl.col("first_peak_hour") == h)).height for h in hours]
    other = [op.filter(~pl.col("high") & (pl.col("first_peak_hour") == h)).height for h in hours]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.2), layout="constrained")
    ax.bar(hours, other, color="#b8c4cc", label=f"Other op. ({sum(other)})")
    ax.bar(hours, high, bottom=other, color=COLORS["C-BAT"], label=f"High peak ({sum(high)})")
    for h in (8, 11):
        ax.annotate(f"{h:02}h", (h, other[h] + high[h]), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7)
    ax.set(xlabel="Hour", ylabel="Days", xticks=np.arange(0, 24, 2), xlim=(-.7, 23.7), ylim=(0, max(o + h for o, h in zip(other, high, strict=True)) + 2))
    ax.legend(loc="upper right", fontsize=7, frameon=False)
    return fig


def model_compare_figure(t1: pl.DataFrame) -> Figure:
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 1.9), sharey=True, layout="constrained")
    for ax, (col, title) in zip(axes, (("rmse", "Slot RMSE (kW)"), ("crps", "CRPS (kW)"), ("peak_mae", "Peak MAE (kW)")), strict=True):
        for i, label in enumerate(LABELS):
            value = t1.filter(pl.col("label") == label)[col].item()
            if np.isfinite(value):
                ax.scatter(value, i, color=COLORS[label], s=28, zorder=3)
                ax.annotate(f"{value:.1f}", (value, i), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=7)
        ax.set(title=title, yticks=range(4), yticklabels=LABELS, ylim=(3.6, -.7))
        ax.margins(x=.2)
    return fig


def peak_strata_figure(t2: pl.DataFrame) -> Figure:
    strata = (("all", "All (51)"), ("nonop", "Nonop (14)"), ("op", "Op (37)"), ("high", "High (13)"))
    fig, ax = plt.subplots(figsize=(WIDTH, 2.3), layout="constrained")
    for i, label in enumerate(LABELS):
        values = [t2.filter(pl.col("label") == label)[f"peak_mae_{s}"].item() for s, _ in strata]
        bars = ax.bar(np.arange(4) + (i - 1.5) * .2, values, .19, color=COLORS[label], label=label)
        ax.bar_label(bars, fmt="%.1f", fontsize=6, padding=1)
    ax.set(xticks=range(4), xticklabels=[t for _, t in strata], ylabel="Peak MAE (kW)")
    ax.margins(y=.12)
    ax.legend(ncol=4, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(.5, 1.13))
    return fig


def select_examples(slots: pl.DataFrame, days: pl.DataFrame) -> pl.DataFrame:
    """Per stratum, the date closest to the stratum-median daily C-BAT MAE (seed 0); date breaks ties."""
    errors = (slots.filter((pl.col("model") == CBAT) & pl.col("y_true").is_finite() & pl.col("y_median").is_finite())
              .group_by("date").agg((pl.col("y_true") - pl.col("y_median")).abs().mean().alias("daily_mae"))
              .join(days, on="date", validate="1:1"))
    out = []
    for label, predicate in (("Nonop", pl.col("kind") == 0), ("Op", (pl.col("kind") > 0) & ~pl.col("high")),
                             ("High peak", pl.col("high"))):
        group = errors.filter(predicate)
        assert group.height, label
        out.append(group.with_columns((pl.col("daily_mae") - pl.col("daily_mae").median()).abs().alias("distance"))
                   .sort("distance", "date").head(1).with_columns(pl.lit(label).alias("stratum")))
    return pl.concat(out)


def forecast_figure(slots: pl.DataFrame, examples: pl.DataFrame) -> Figure:
    fig, axes = plt.subplots(3, 1, figsize=(WIDTH, 5.2), sharex=True, layout="constrained")
    color = COLORS["C-BAT"]
    for ax, row in zip(axes, examples.iter_rows(named=True), strict=True):
        day = slots.filter(pl.col("date") == row["date"]).sort("datetime")
        c = day.filter(pl.col("model") == CBAT)
        x = np.arange(c.height) / 4
        ax.fill_between(x, c["q05"], c["q95"], color=color, alpha=.15, lw=0, label="C-BAT 90%")
        ax.fill_between(x, c["q25"], c["q75"], color=color, alpha=.3, lw=0, label="C-BAT 50%")
        for label, style in (("BAT", "--"), ("LightGBM", "-.")):
            ax.plot(x, day.filter(pl.col("model") == SLOT_MODELS[label])["y_median"], style, color=COLORS[label], lw=.8, label=label)
        ax.plot(x, c["y_median"], color=color, lw=1.2, label="C-BAT")
        ax.plot(x, c["y_true"], color="black", lw=1, label="Observed")
        ax.set(title=f"{row['stratum']} · {row['date'][5:].replace('.', '-')} · MAE {row['daily_mae']:.1f}", ylabel="kW", xlim=(0, 24))
    axes[-1].set(xticks=np.arange(0, 25, 4), xlabel="Hour")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=6, fontsize=7, frameon=False)
    return fig


def noise_figure(traces: list[pl.DataFrame]) -> Figure:
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.1), layout="constrained")
    for ax, (param, title) in zip(axes, (("sigma_u_kw", "Day SD (kW)"), ("sigma_eta_kw", "Innov. SD (kW)"), ("rho", "AR ρ")), strict=True):
        data = [np.concatenate([t[f"{param}[{g}]"].to_numpy() for t in traces]) for g in (0, 1)]
        boxes = ax.boxplot(data, tick_labels=["Nonop", "Op"], widths=.5, patch_artist=True, showfliers=False,
                           medianprops={"color": "black", "linewidth": .8})
        for box, c in zip(boxes["boxes"], (COLORS["B0_kind"], COLORS["C-BAT"]), strict=True):
            box.set(facecolor=c, alpha=.6)
        ax.set_title(title)
    return fig




def generate(out: Path = ROOT / "report/figs") -> list[Path]:
    inputs: dict[str, str] = {}

    def read(path: Path) -> pl.DataFrame:
        inputs[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return pl.read_csv(path)

    days = read(R / "peak_conditions_v2/observed_days.csv")
    t1 = read(R / "report_tables/t1_main.csv").with_columns(pl.col("label").map_elements(table_label, return_dtype=pl.String))
    t2 = read(R / "report_tables/t2_peak_strata.csv").with_columns(pl.col("label").map_elements(table_label, return_dtype=pl.String))
    slots = pl.concat([read(R / "attention_ablation/seed0/oof_slots.csv").filter(pl.col("model") == CBAT).drop("seed"),
                       read(R / "robustness/oof_slots.csv").filter(pl.col("model").is_in(["BAT", "M2_tuned"]))])
    assert slots["date"].max() < "2021.09.01" and days["date"].max() < "2021.09.01"
    traces = [read(p) for p in sorted((R / "attention_ablation/jobs").glob(f"seed*_f*_{CBAT}/posterior_trace.csv"))]
    assert len(traces) == 12, len(traces)
    examples = select_examples(slots, days)
    out.mkdir(parents=True, exist_ok=True)
    figures = {"peak_hours": lambda: peak_hours_figure(days), "model_compare": lambda: model_compare_figure(t1),
               "peak_strata": lambda: peak_strata_figure(t2), "forecast_examples": lambda: forecast_figure(slots, examples),
               "noise_state": lambda: noise_figure(traces)}
    paths = []
    with plt.rc_context(STYLE):
        for name, make in figures.items():
            fig = make()
            path = out / f"{name}.pdf"
            fig.savefig(path, metadata={"Creator": "gmst.report_figures", "CreationDate": None, "ModDate": None})
            plt.close(fig)
            paths.append(path)
    manifest = {"input_sha256": inputs, "pdfs": [p.name for p in paths],
                "examples": examples.select("stratum", "date", "fold", "daily_mae", "M_true", "C90").to_dicts(),
                "example_seed": 0, "example_model": CBAT,
                "example_selection": "Closest to stratum-median daily slot MAE (finite pairs, usable peak days); date breaks ties",
                "noise_state": "Posterior draws pooled over folds f1-f4 and seeds 0-2; boxes 25/50/75%, whiskers 1.5 IQR, outliers hidden",
                "refitted": False, "september_opened": False}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return paths


if __name__ == "__main__":
    for p in generate():
        print(p)
