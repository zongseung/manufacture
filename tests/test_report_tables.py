from datetime import date, timedelta

import numpy as np

from gmst.report_tables import paired_ci


def test_paired_ci_constant_difference_and_calendar_gaps() -> None:
    calendar = [date(2021, 7, 1) + timedelta(days=i) for i in range(20)]
    dates = calendar[::2]
    out = paired_ci(np.full(10, 3.0), np.full(10, 1.5), dates, calendar)
    assert out["n_days"] == 10
    for key in ("diff", "ci_lo", "ci_hi", "block7_lo", "block7_hi"):
        assert np.isclose(out[key], 2.0), key
