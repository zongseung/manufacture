import numpy as np
import pytest
from test_backbone import synthetic_panel

from gmst import benchmark_models as bm


def test_kind_reference_prefers_same_kind_and_recent_daytype() -> None:
    # Given all days operating, but only two previous plans match the target type.
    p = synthetic_panel(45)
    p["X"]["생산량"] = np.ones((45, 96))
    p["X"]["생산량"][[10, 38, 44], 40:] = 0
    p["dtype"][:] = 0
    p["dtype"][38] = 1
    # When a recent same-daytype match is absent, an older same-kind day is allowed.
    ref, status = bm.b0_kind_ref(p, 44)
    # Then the most recent same kind wins over a more recent different kind.
    assert (ref, status) == (38, "same_kind")
    p["dtype"][38] = 0
    assert bm.b0_kind_ref(p, 44) == (38, "same_kind_daytype_28d")
    p["Y"][38, 0] = np.nan
    assert bm.b0_kind_ref(p, 44) == (10, "same_kind")


def test_kind_reference_fallback_and_predictions_are_past_only() -> None:
    # Given a target type with no complete previous match.
    p = synthetic_panel(40)
    p["X"]["생산량"] = np.ones((40, 96))
    p["X"]["생산량"][35, 40:] = 0
    p["Y"][34, 0] = np.nan
    model = bm.b0_kind_model()
    state = model["fit"](p, "f1", np.array([100., 120., 140.]))
    # When issuing a forecast with mutated current and future targets.
    p["Y"][35:] = 9999
    pred = model["predict"](state, p, 35)
    # Then fallback is the latest complete past curve and the empty-history case is explicit.
    assert bm.b0_kind_ref(p, 35) == (33, "latest_complete")
    np.testing.assert_array_equal(pred["y_median"], p["Y"][33])
    assert bm.b0_kind_ref(p, 0) == (None, "no_complete_history")
    assert np.isnan(model["predict"](state, p, 0)["y_median"]).all()


def test_grid_includes_legacy_and_meaningful_capacity_choices() -> None:
    # Given the registered full tuning grid.
    grid = bm.candidate_grid()
    # When examining the candidates.
    capacities = {(c.num_leaves, c.min_data_in_leaf, c.learning_rate, c.rounds) for c in grid}
    # Then the old default and requested capacity/rate/round alternatives are included.
    assert (31, 20, .1, 100) in capacities
    assert len(capacities) == 37
    assert (7, 10, .03, 100) in capacities and (31, 60, .1, 300) in capacities
    daily = bm.daily_candidate_grid()
    assert bm.Candidate() in daily and len(daily) == 37
    assert bm.Candidate(3, 2, .03, 100) in daily
    assert bm.Candidate(15, 10, .1, 300) in daily


def test_tuning_uses_only_fit_and_tune_and_integrates_with_evaluator(monkeypatch: pytest.MonkeyPatch) -> None:
    from gmst import evaluate as ev
    from gmst.features import internal_split

    # Given a tiny real LightGBM grid and disjoint train/tune/calibration/outer days.
    monkeypatch.setattr(bm, "candidate_grid", lambda quick=False: (
        bm.Candidate(7, 10, .1, 2), bm.Candidate(15, 30, .03, 3)))
    monkeypatch.setattr(bm, "daily_candidate_grid", lambda quick=False: (
        bm.Candidate(3, 2, .1, 4), bm.Candidate(5, 3, .03, 5)))
    p = synthetic_panel(35)
    model = bm.tuned_model(10., 60)
    C = np.array([100., 120., 140.])
    state = model["fit"](p, "f1", C)
    _, tune, cal = internal_split(p, "f1")
    # When calibration and outer observations change.
    p["Y"][cal[0]:] += 9000
    altered = model["fit"](p, "f1", C)
    slots, days, _, inner_rows = ev.rolling_origin(model, p, folds=("f1",), states={"f1": state})
    # Then tuning and the inner fit stay identical, while the evaluator accepts all heads.
    assert bm.tuning_table(state).equals(bm.tuning_table(altered))
    inner = model["inner_state"](state)
    altered_inner = model["inner_state"](altered)
    assert inner is not None and altered_inner is not None
    assert inner.model["train_idx"].max() < cal.min()
    assert inner.model["mean"].model_to_string() == altered_inner.model["mean"].model_to_string()
    assert inner.model["Mq"][9].model_to_string() == altered_inner.model["Mq"][9].model_to_string()
    assert len(state.model["names"]) == 32
    assert slots.height == 5 * 96 and days.height == 5 and inner_rows.height == len(cal)
    audit = bm.tuning_table(state)
    assert audit["n_tune_days"].to_list() == [len(tune)] * 4
    for target, heads in (("slot", [state.model["mean"], *state.model["q"]]),
                          ("peak", [*state.model["Mq"], *state.model["clf"]])):
        target_audit = audit.filter(audit["target"] == target)
        assert target_audit["selected"].sum() == 1
        selected = target_audit.filter(target_audit["selected"]).row(0, named=True)
        assert selected["tune_mae"] == target_audit["tune_mae"].min()
        for head in heads:
            if head is not None:
                assert head.params["num_leaves"] == selected["num_leaves"]
                assert head.params["min_data_in_leaf"] == selected["min_data_in_leaf"]
                assert head.params["num_iterations"] == selected["rounds"]


def test_tuned_model_rejects_final_fold() -> None:
    # Given a development-only model.
    model = bm.tuned_model(10., 60, quick=True)
    # When asked to fit the sealed final fold, then it rejects before reading targets.
    with pytest.raises(ValueError, match="development folds"):
        model["fit"](synthetic_panel(), "test", np.array([100., 120., 140.]))
