"""One command, raw data → submission: fit the final C-BAT on data before 2021-09-01 and write
results_v3/submission/test_predictions.csv (+ eval_mask.csv, test metrics, posterior tables)."""
import os

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(name, "1")

from gmst import ROOT  # noqa: E402
from gmst.conditional_bat import ConditionalConfig  # noqa: E402
from gmst.run_conditional import run_final  # noqa: E402

if __name__ == "__main__":
    print(run_final(ROOT / "results_v3/submission", ConditionalConfig(n_iter=6000, burn=3000, seed=0)))
