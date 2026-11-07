"""Predictor contract.

    fit(train)                 offline, sees only training benchmarks (any library allowed)
    adapt(labeled) -> state    pure function of the revealed evidence
    extend(state, new)         optional: state for labeled + new, given state for labeled
    score(state, s, i) -> p    one probability
    score_batch(state, inputs) vectorised version; must agree with score

`predict(input, labeled)` is the official signature and is derived. Runtime code paths
(adapt/extend/score) must use only numpy + stdlib so the fitted object can ship in the ZIP.
Hosted workers restart at each checkpoint, so nothing may depend on state left by earlier
calls except caches that are pure functions of the evidence.

Evidence caching: states are keyed by a sampled fingerprint of `labeled` (length plus a
cheap hash of up to ~48 entries). Evidence only grows during a run, so when a new list
extends the last one seen, the state is extended incrementally instead of rebuilt.
"""
from collections import defaultdict

import numpy as np

from ..features.subject import subject_key


def item_key(item: dict) -> tuple:
    return (item.get("benchmark_id", ""), item.get("item_features", ""), item.get("interactors", ""),
            item.get("item_content", ""))


def _entry_fp(entry) -> int:
    (s, i), y = entry
    return hash((y, i.get("benchmark_id", ""), i.get("item_content", ""), i.get("item_features", ""),
                 i.get("interactors", ""), s.get("normalized_name", ""), s.get("subject_features_extra", ""),
                 s.get("provider", ""), s.get("harness", ""), s.get("harness_version", ""),
                 s.get("reasoning_effort", ""), s.get("release_date", ""), s.get("access_date", "")))


def fingerprint(labeled, n=None) -> tuple:
    """Fingerprint of labeled[:n]: length + entries at sampled positions (all if short)."""
    n = len(labeled) if n is None else n
    idx = range(n) if n <= 48 else sorted(set(range(0, n, n // 32)) | set(range(n - 16, n)))
    return (n, tuple(_entry_fp(labeled[k]) for k in idx))


class Evidence:
    """Revealed labels grouped for O(1) lookups.

    by_bench[b]       list of (subject_key, item_key, y, subject, item)
    pair[(b, skey)]   [successes, count] for one subject-benchmark pair
    bench[b]          [successes, count] over all subjects on benchmark b
    """

    def __init__(self, labeled=()):
        self.by_bench, self.pair, self.bench = defaultdict(list), {}, {}
        self.n_pairs = defaultdict(int)   # distinct subjects with labels, per benchmark
        self.add(labeled)

    def add(self, labeled) -> set:
        touched = set()
        for (s, i), y in labeled:
            b, sk, y = i.get("benchmark_id", ""), subject_key(s), float(y)
            self.by_bench[b].append((sk, item_key(i), y, s, i))
            if (b, sk) not in self.pair:
                self.n_pairs[b] += 1
            for c in (self.pair.setdefault((b, sk), [0.0, 0]), self.bench.setdefault(b, [0.0, 0])):
                c[0] += y
                c[1] += 1
            touched.add(b)
        return touched

    def pair_counts(self, subject, item):
        c = self.pair.get((item.get("benchmark_id", ""), subject_key(subject)))
        return (c[0], c[1]) if c else (0.0, 0)

    def bench_counts(self, item):
        c = self.bench.get(item.get("benchmark_id", ""))
        return (c[0], c[1]) if c else (0.0, 0)


class Predictor:
    name = "base"
    clip = (0.0, 1.0)

    def __init__(self, **params):
        for k, v in params.items():
            if not hasattr(type(self), k):
                raise ValueError(f"{type(self).__name__} has no parameter {k!r}")
            setattr(self, k, tuple(v) if isinstance(v, list) else v)
        self._cache = {}

    def fit(self, train) -> "Predictor":
        return self

    def adapt(self, labeled):
        return Evidence(labeled)

    def extend(self, state, new):
        state.add(new)
        return state

    def score(self, state, subject, item) -> float:
        raise NotImplementedError

    def score_batch(self, state, inputs) -> np.ndarray:
        return np.array([self.score(state, s, i) for s, i in inputs], float)

    # ---- derived ----
    def _state(self, labeled):
        labeled = labeled or []
        fp = fingerprint(labeled)
        if fp in self._cache:
            return self._cache[fp]
        last = getattr(self, "_cache_last", None)
        if last and 0 < last[0][0] < len(labeled) and fingerprint(labeled, last[0][0]) == last[0]:
            state = self.extend(last[1], labeled[last[0][0]:])  # mutates: retire the old key
            self._cache.pop(last[0], None)
        else:
            state = self.adapt(labeled)
        if len(self._cache) > 64:
            self._cache.clear()
        self._cache[fp] = state
        self._cache_last = (fp, state)
        return state

    def _finish(self, p):
        lo, hi = self.clip
        return np.clip(np.nan_to_num(p, nan=0.5), lo, hi)

    def predict(self, input, labeled=None) -> float:
        subject, item = input
        return float(self._finish(self.score(self._state(labeled), subject, item)))

    def predict_batch(self, inputs, labeled=None) -> np.ndarray:
        return self._finish(self.score_batch(self._state(labeled), inputs))

    def posterior_parts(self, input, labeled=None):
        """Posterior components of one input's logit (see IRT.parts_batch), or None if the
        predictor does not expose them. Uses the same evidence-keyed state cache as predict."""
        if not hasattr(self, "parts_batch"):
            return None
        out = self.parts_batch(self._state(labeled), [tuple(input)])
        return {k: float(v[0]) for k, v in out.items()}

    def reset_runtime(self) -> None:
        """Drop all caches (simulates the hosted worker restart at each checkpoint)."""
        for k in [k for k in self.__dict__ if k.startswith("_cache")]:
            delattr(self, k)
        self._cache = {}

    def __getstate__(self):
        d = {k: v for k, v in self.__dict__.items() if not k.startswith("_cache")}
        d["_cache"] = {}
        return d
