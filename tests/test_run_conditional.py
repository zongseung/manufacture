import numpy as np
import polars as pl
import pytest

from gmst.conditional_bat import ConditionalConfig, conditional_model
from gmst.conditional_diagnostics import basic_split_rhat, peak_interval_summary
from gmst.run_conditional import NOISE_VARIANTS, verify_alignment


def test_alignment_rejects_changed_truth_and_peak_eligibility() -> None:
    slots = pl.DataFrame({"model": ["BAT", "BAT"], "fold": ["f1", "f1"], "date": ["2021.07.07"] * 2,
                          "datetime": ["2021.07.07 00:00:00", "2021.07.07 00:15:00"],
                          "y_true": [10., np.nan], "is_missing": [False, True]})
    days = pl.DataFrame({"model": ["BAT"], "fold": ["f1"], "date": ["2021.07.07"],
                         "usable_peak": [False], "n_obs": [1], "peak_slot_true": [0],
                         "M_true": [10.], "C50": [12.], "C75": [15.], "C90": [19.],
                         "event_C50": [0.], "event_C75": [0.], "event_C90": [0.]})
    candidate_slots = slots.with_columns(pl.lit("candidate").alias("model"))
    candidate_days = days.with_columns(pl.lit("candidate").alias("model"))
    verify_alignment(candidate_slots, candidate_days, slots, days)
    with pytest.raises(AssertionError):
        verify_alignment(candidate_slots.with_columns(pl.lit(10.).alias("y_true")), candidate_days, slots, days)
    with pytest.raises(AssertionError):
        verify_alignment(candidate_slots, candidate_days.with_columns(pl.lit(True).alias("usable_peak")), slots, days)


def test_basic_split_rhat_flags_different_chain_locations() -> None:
    chains = np.random.default_rng(10).normal(size=(3, 1000, 2))
    chains[:, :, 1] += np.arange(3)[:, None] * 5
    values = basic_split_rhat(chains)
    assert values[0] < 1.01
    assert values[1] > 2
    assert np.isnan(basic_split_rhat(np.zeros((3, 1000, 1)))[0])


def test_peak_interval_summary_excludes_unusable_days() -> None:
    frame = pl.DataFrame({"model": ["conditional"] * 3, "op": [0, 1, 1],
                          "usable_peak": [True, True, False], "C90": [20.] * 3,
                          "M_true": [10., 25., 999.], "q05": [7.] * 3, "q10": [8.] * 3,
                          "q25": [9.] * 3, "q50": [10.] * 3, "q75": [11.] * 3,
                          "q90": [30.] * 3, "q95": [40.] * 3})
    summary = peak_interval_summary(frame)
    all_days = summary.filter(pl.col("stratum") == "all").row(0, named=True)
    assert all_days["n_days"] == 2
    assert all_days["coverage50"] == .5
    assert all_days["coverage80"] == 1.
    assert all_days["width90_kw"] == 33.
    assert summary.filter(pl.col("stratum") == "high")["n_days"].item() == 1


def test_final_scores_test_fold_and_records_opened_september(tmp_path, monkeypatch) -> None:
    import json

    from gmst import features
    from gmst import run_conditional as rc
    from gmst.conditional_bat import ConditionalConfig

    load = features.load_panel
    requested = []
    # Keep September sealed in tests: record the unseal request but serve the sealed panel.
    monkeypatch.setattr(features, "load_panel", lambda **kw: requested.append(kw) or load())
    rc.run_final(tmp_path, ConditionalConfig(n_iter=20, burn=10))
    assert requested == [{"unseal": True}]
    status = json.loads((tmp_path / "run_status.json").read_text())
    assert status["september_opened"] is True and status["status"] == "complete"
    assert status["model"] == "BAT_conditional_gaussian_fixed_attention"
    slots = pl.read_csv(tmp_path / "test_slots.csv")
    assert slots.height == 14 * 96 and set(slots["fold"]) == {"test"}
    assert slots["date"].min() == "2021.09.01" and slots["date"].max() == "2021.09.14"
    assert (tmp_path / "posterior_summary.csv").exists()
    preds = pl.read_csv(tmp_path / "test_predictions.csv")
    reference = ["datetime", "model", "y_mean", "y_median", *(f"q{k:02d}" for k in range(5, 100, 5)),
                 "M_hat_median", "M_hat_mean", "peak_time_mode", "risk_C50", "risk_C75", "risk_C90",
                 "C50", "C75", "C90"]
    assert preds.height == 14 * 96 and preds.columns == reference
    assert set(preds["model"]) == {"BAT_conditional_gaussian_fixed_attention"}
    assert preds["datetime"][0] == "2021.09.01 00:00:00" and preds["peak_time_mode"].str.contains(r"^\d\d:\d\d$").all()
    assert pl.read_csv(tmp_path / "eval_mask.csv").columns == ["datetime", "is_missing"]


def test_noise_variants_start_with_default_control() -> None:
    configs = [ConditionalConfig(noise_state=s, innovation=i) for s, i in NOISE_VARIANTS]
    assert configs[0] == ConditionalConfig()
    base = "BAT_conditional_gaussian_fixed_attention"
    assert [conditional_model(c)["name"] for c in configs] == [
        base, base + "_noisekind", base + "_noiseslot", base + "_tinnov", base + "_noisekind_tinnov"]


def test_final_outputs_can_issue_raw_path_risk_instead_of_platt(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from datetime import datetime, timedelta

    import polars as pl

    from gmst import evaluate as ev
    from gmst.run_v3 import _final_outputs

    stamps = [datetime(2021, 9, 1) + timedelta(minutes=15 * i) for i in range(14 * 96)]
    slots = pl.DataFrame({"datetime": stamps, "model": "BAT", "y_true": 1., "y_mean": 1., "y_median": 1.,
                          **{q: 1. for q in ev.QCOLS}}).with_columns(date=pl.col("datetime").dt.date())
    days = pl.DataFrame({"date": sorted(set(slots["date"])), "model": "BAT", "M_hat_median": 1., "M_hat_mean": 1.,
                         "peak_time_mode": 40, **{c: 1. for c in ev.CS},
                         **{f"risk_raw_{c}": .7 for c in ev.CS}, **{f"risk_platt_{c}": .1 for c in ev.CS}})
    _final_outputs(tmp_path, slots, days, risk="raw")
    issued = pl.read_csv(tmp_path / "test_predictions.csv")
    assert (issued["risk_C90"] == .7).all()
    _final_outputs(tmp_path, slots, days)
    assert (pl.read_csv(tmp_path / "test_predictions.csv")["risk_C90"] == .1).all()


def test_comparison_table_builds_the_same_configs_as_the_original_flags() -> None:
    from gmst.conditional_bat import ConditionalConfig
    from gmst.run_conditional import COMPARISONS, NOISE_VARIANTS, configs

    base = ConditionalConfig(n_iter=10, burn=5, seed=1)
    as_set = lambda name: {c for c in configs(name, base)}  # noqa: E731
    assert as_set(None) == {base}
    assert as_set("attention_ablation") == {base, replace_(base, fixed_attention=False)}
    assert as_set("pooled_noise_comparison") == {replace_(base, fixed_attention=False),
                                                 replace_(base, conditional=False, fixed_attention=False)}
    assert as_set("ref_comparison") == {base, replace_(base, ref="kind")}
    assert len(as_set("noise_comparison")) == len(NOISE_VARIANTS)
    assert {folder for folder, _, _ in COMPARISONS.values()} >= {"attention_ablation", "noise_ablation", "ref_ablation"}


def replace_(config, **changes):  # type: ignore[no-untyped-def]
    from dataclasses import replace

    return replace(config, **changes)
