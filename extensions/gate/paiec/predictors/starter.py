"""Starter model: a short, commented template to copy when writing a new predictor.

    p(subject, item) = (k + alpha * mu) / (n + alpha)
    mu = sigmoid(benchmark log-odds + subject's training-history offset)

k / n are the pair's revealed successes / labels. The benchmark log-odds start at the training base
rate and move toward the pooled rate of every label revealed on this (unseen) benchmark so far.
Copy this file, rename the class and `name`, register it in predictors/__init__.py, and point a config
at it (configs/aims/starter.yaml). See docs/getting-started.md. As written it is the `beta_shrink`
baseline (identical scores), so an unmodified copy should reproduce that row of the README table.
"""
import numpy as np

from ..features.subject import subject_identity
from .base import Predictor
from .simple import logit, sigmoid, subject_history


class Starter(Predictor):
    name = "starter"          # the config's `predictor.name`
    # Every public class attribute is a hyperparameter: override with `--set predictor.alpha=8`.
    alpha = 4.0               # pseudo-counts of the prior when mixing in the pair's own labels
    bench_alpha = 10.0        # pseudo-counts of the base rate when pooling labels on the benchmark
    history_weight = 1.0      # 0 ignores what the subject did on training benchmarks
    # Fitted state (set by fit). It is pickled into the submission ZIP, so keep it to plain
    # Python / numpy objects.
    base_rate = 0.5
    history = None            # canonical model identity -> (mean logit offset, #benchmarks)

    def fit(self, train):
        """Offline, on training benchmarks only (`train` is a paiec.data.dataset.Dataset).
        Any library may be used here; only score-time code must stay numpy + stdlib."""
        self.base_rate = float(train.resp_y.mean())
        self.history = subject_history(train)
        return self

    # adapt(labeled) -> state is inherited: an `Evidence` object with O(1) per-pair and per-benchmark
    # counts of the revealed labels. Override adapt/extend if your state is more than counts.

    def score(self, ev, subject, item):
        """Called at test time with the visible subject / item dicts (all strings)."""
        K, N = ev.bench_counts(item)                       # every revealed label on this benchmark
        bench = (K + self.bench_alpha * self.base_rate) / (N + self.bench_alpha)
        off, c = (self.history or {}).get(subject_identity(subject), (0.0, 0))
        mu = float(sigmoid(logit(bench) + self.history_weight * off * c / (c + 1)))
        k, n = ev.pair_counts(subject, item)               # this subject's labels on this benchmark
        return (k + self.alpha * mu) / (n + self.alpha)

    def score_batch(self, ev, inputs):
        # Optional speed-up; the default loops over score. Tests check the two agree.
        return np.array([self.score(ev, s, i) for s, i in inputs], float)
