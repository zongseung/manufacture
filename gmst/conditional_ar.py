from dataclasses import dataclass

import numpy as np

from gmst.contracts import FloatArray, IntArray


@dataclass(frozen=True, slots=True)
class ObservationGrid:
    day: IntArray
    slot: IntArray
    previous: IntArray
    gap: IntArray

    @classmethod
    def from_targets(cls, targets: FloatArray) -> "ObservationGrid":
        day, slot = np.nonzero(np.isfinite(targets))
        previous = np.maximum(np.arange(len(day)) - 1, 0)
        first = (np.arange(len(day)) == 0) | (day != day[previous])
        previous[first] = np.flatnonzero(first)
        return cls(day, slot, previous, slot - slot[previous])

    def weights(self, rho: FloatArray) -> tuple[FloatArray, FloatArray]:
        r = rho[self.day]
        phi = np.where(self.gap > 0, r ** self.gap, 0.)
        return phi, np.sqrt((1 - r * r) / (1 - phi * phi))

    def transition_variance(self, rho: FloatArray, var: FloatArray) -> FloatArray:
        """Var(e_t − ρ^s e_{t−s}) = Σ_{j<s} ρ^{2j} var[d, t−j] for e_t = ρ e_{t−1} + √var[d, t] η_t.

        rho is per day, var is (days, 96); a day's first observation is stationary, var[d, t]/(1 − ρ²).
        """
        r = rho[self.day]
        total = var[self.day, self.slot]
        long = np.flatnonzero(self.gap > 1)
        for j in range(1, self.gap.max(initial=0)):
            long = long[self.gap[long] > j]
            total[long] += r[long] ** (2 * j) * var[self.day[long], self.slot[long] - j]
        first = self.gap == 0
        total[first] /= 1 - r[first] ** 2
        return total

    def whiten(self, values: FloatArray, weights: tuple[FloatArray, FloatArray]) -> FloatArray:
        phi, inv_sd = weights
        return (values - phi * values[self.previous]) * inv_sd


@dataclass(frozen=True, slots=True)
class SparseDesign:
    columns: IntArray
    values: FloatArray

    def rhs(self, weights: FloatArray, target: FloatArray) -> FloatArray:
        return sum((np.bincount(self.columns[:, i], weights * self.values[:, i] * target,
                                minlength=97) for i in range(3)), start=np.zeros(97))

    def normal_equations(self, weights: FloatArray, target: FloatArray) -> tuple[FloatArray, FloatArray]:
        precision = np.zeros((97, 97))
        for i in range(3):
            ci, vi = self.columns[:, i], self.values[:, i]
            for j in range(3):
                np.add.at(precision, (ci, self.columns[:, j]), weights * vi * self.values[:, j])
        return precision, self.rhs(weights, target)
