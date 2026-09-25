import numpy as np
import polars as pl
import pytest

from gmst.conditional_diagnostics import basic_split_rhat, peak_interval_summary
from gmst.run_conditional import verify_alignment


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
