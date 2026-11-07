"""Baselines and empirical-Bayes shrinkage."""
from collections import defaultdict

import numpy as np

from ..features.subject import subject_identity
from .base import Predictor


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def subject_history(train) -> dict:
    """identity -> (mean logit offset vs. benchmark base rate, #benchmarks) on training data."""
    offs = defaultdict(list)
    b = train.resp_b
    for bb in np.unique(b):
        m = b == bb
        # Laplace-smoothed rates: a 50/50 or 0/50 pair would otherwise clip to +-9 logits whatever its size
        base = logit((train.resp_y[m].sum() + 0.5) / (m.sum() + 1))
        sums, cnt = np.bincount(train.resp_s[m], train.resp_y[m]), np.bincount(train.resp_s[m])
        for s in np.nonzero(cnt)[0]:
            offs[subject_identity(train.subjects[s])].append(float(logit((sums[s] + 0.5) / (cnt[s] + 1)) - base))
    return {k: (float(np.mean(v)), len(v)) for k, v in offs.items()}


class Constant(Predictor):
    name = "constant"
    p = 0.5

    def score(self, ev, subject, item):
        return self.p


class EmpiricalMean(Predictor):
    """Official baseline: mean of the pair's revealed labels, else `prior`."""
    name = "empirical_mean"
    prior = 0.5

    def score(self, ev, subject, item):
        k, n = ev.pair_counts(subject, item)
        return k / n if n else self.prior


class BetaShrink(Predictor):
    """p = (k + alpha * mu) / (n + alpha), with the prior mean mu built from switches:

    prior_const     fallback base rate (set to the training mean if fit_const)
    use_benchmark   pooled rate of all revealed labels on this benchmark, shrunk to the base
                    rate with bench_alpha (only helps when evidence is global)
    use_history     subject's mean logit offset vs. benchmark base rates on training benchmarks
    """
    name = "beta_shrink"
    alpha = 4.0
    prior_const = 0.5
    fit_const = True
    use_benchmark = True
    bench_alpha = 10.0
    use_history = True
    history_strength = 1.0
    history = None   # identity -> (logit offset, #benchmarks); set by fit

    def fit(self, train):
        if self.fit_const and len(train.resp_y):
            self.prior_const = float(train.resp_y.mean())
        self.history = subject_history(train) if self.use_history else {}
        return self

    def prior_mean(self, ev, subject, item):
        mu = self.prior_const
        if self.use_benchmark:
            K, N = ev.bench_counts(item)
            mu = (K + self.bench_alpha * mu) / (N + self.bench_alpha)
        if self.use_history:
            off, c = (self.history or {}).get(subject_identity(subject), (0.0, 0))
            mu = float(sigmoid(logit(mu) + off * c / (c + self.history_strength)))
        return mu

    def score(self, ev, subject, item):
        k, n = ev.pair_counts(subject, item)
        return (k + self.alpha * self.prior_mean(ev, subject, item)) / (n + self.alpha)
