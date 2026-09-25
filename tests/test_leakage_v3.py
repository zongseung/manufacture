import numpy as np

from gmst.leakage_v3 import build_folds


def test_folds_cover_days_and_keep_events_whole() -> None:
    event = ["E0", "E1", "E1", "E2", "E1", "E3", "E3", "E4", "E5", "E6", "E0", "E7"]
    idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11], dtype=np.int64)  # day 9 not evaluable
    folds = build_folds(idx, event)
    random = np.concatenate(folds["random5"])
    assert len(folds["random5"]) == 5 and np.array_equal(np.sort(random), idx)
    loeo = folds["loeo"]
    assert np.array_equal(np.sort(np.concatenate(loeo)), idx)
    assert len(loeo) == len({event[d] for d in idx}) == 7  # E6 has no evaluable day → no fold
    for f in loeo:
        assert len({event[d] for d in f}) == 1
        assert set(f) == {d for d in idx if event[d] == event[f[0]]}
