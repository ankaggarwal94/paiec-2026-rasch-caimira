"""Competition metrics. See docs/evaluation.md."""
import numpy as np

BUDGETS = (0, 1, 3, 7, 15, 31)
# Trapezoid over x = log2(n+1) = 0..5, normalised by 5 -> 0.1,0.2,0.2,0.2,0.2,0.1
ALC_WEIGHTS = (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)


def brier(p, y) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def ece(p, y, n_bins: int = 10) -> float:
    """Expected calibration error, equal-width bins, weighted by count (diagnostic only)."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def brier_alc(brier_by_budget: dict) -> float:
    """Lower is better. Perfect=0; constant 0.5 -> 0.25."""
    return float(sum(w * brier_by_budget[b] for w, b in zip(ALC_WEIGHTS, BUDGETS)))
