"""Label-count-gated logit blend of several test-time variants of one fitted model (irt_item_lr's fit).

    logit p(s, i) = sum_m w_m[bin(n_own)] * logit p_m(s, i) + c[bin(n_own)]

- Members are runtime configurations of IRTItemLR (e.g. {lr_on: false} = item amortisation A3,
  {lr_on: true} = item + subject-descriptor regression, {vi: true} = mean-field VI). They share one fitted
  object (FIT_AS = IRTItemLR), so a member costs one more per-benchmark adaptation at test time, no fit.
- n_own = labels the pair (subject, benchmark) already has in `labeled`; bins 0, 1, 2-3, 4-7, 8-15, 16+
  (budgets 0, 1, 3, 7, 15, 31 on the replay). The hosted predict() sees the same quantity.
- Weights are fitted offline on dev replays (analysis/auto/blend.py). `mix_fold_w` maps a dev benchmark
  name to the weights fitted *without* its fold (leave-one-fold-out), so dev runs are honest; on hosted
  (anonymous benchmarks) `mix_w` (all-fold fit) is used.
Runtime: numpy + stdlib; adapt / extend / score are pure functions of `labeled`.
"""
from contextlib import contextmanager

import numpy as np

from ..data.dataset import anon_benchmark_id
from .base import Evidence
from .irt_item import IRTItem, IRTItemLR
from .simple import sigmoid

EDGES = (0, 1, 2, 4, 8, 16)   # bin k: EDGES[k] <= n_own < EDGES[k + 1]


class IRTItemMix(IRTItemLR):
    name = "irt_item_mix"
    FIT_AS = IRTItemLR
    RUNTIME_PARAMS = IRTItemLR.RUNTIME_PARAMS + ("mix_members", "mix_w", "mix_c", "mix_fold_w", "mix_fold_c",
                                               "emb_online")
    mix_members = ({"lr_on": False}, {"lr_on": True})
    mix_w = ((1.0,) * 6, (0.0,) * 6)     # [member][bin]
    mix_c = (0.0,) * 6                   # [bin]
    mix_fold_w = None                    # {benchmark: [member][bin]} (dev evaluation only)
    mix_fold_c = None                    # {benchmark: [bin]}
    # DEV-ONLY probe: frozen mpnet item embeddings (features/mpnet.py) as the online content-map features instead of
    # the hashed ItemFeaturizer vector. Not hosted-safe: unseen hosted items would need the torch encoder at runtime.
    emb_online = False
    _EMB = {}

    def online_features(self, items, X):
        if not self.emb_online:
            return X
        if "t" not in IRTItemMix._EMB:   # process-wide (survives reset_runtime; never pickled)
            from ..features.mpnet import EmbeddingTable
            IRTItemMix._EMB["t"] = EmbeddingTable()
        return IRTItemMix._EMB["t"].lookup(items)

    @staticmethod
    def _lg(p):
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p))

    @contextmanager
    def _member(self, k):
        ov = dict(self.mix_members[k])
        old = {a: self.__dict__[a] for a in ov if a in self.__dict__}
        for a, v in ov.items():
            setattr(self, a, v)
        try:
            yield
        finally:
            for a in ov:
                if a in old:
                    setattr(self, a, old[a])
                else:
                    self.__dict__.pop(a, None)

    def adapt(self, labeled):
        out = []
        for k in range(len(self.mix_members)):
            with self._member(k):
                out.append(IRTItem.extend(self, {"ev": Evidence(), "fits": {}, "joint": None}, labeled))
        return {"m": out}

    def extend(self, state, new):
        for k, st in enumerate(state["m"]):
            with self._member(k):
                state["m"][k] = IRTItem.extend(self, st, new)
        return state

    def _by_anon(self, table):
        """{benchmark name: v} -> {anonymised benchmark_id: v}: items only carry the anonymised id."""
        cache = self.__dict__.setdefault("_cache_fold", {})
        if id(table) not in cache:
            cache[id(table)] = {anon_benchmark_id(k): v for k, v in (table or {}).items()}
        return cache[id(table)]

    def _weights(self, bench):
        W = self._by_anon(self.mix_fold_w).get(bench, self.mix_w)
        c = self._by_anon(self.mix_fold_c).get(bench, self.mix_c)
        return np.asarray(W, float), np.asarray(c, float)

    def score_batch(self, state, inputs):
        L = []
        for k, st in enumerate(state["m"]):
            with self._member(k):
                L.append(self._lg(IRTItem.score_batch(self, st, inputs)))
        L = np.stack(L)
        ev = state["m"][0]["ev"]
        out = np.empty(len(inputs))
        for j, (s, i) in enumerate(inputs):
            n = ev.pair_counts(s, i)[1]
            b = int(np.searchsorted(EDGES, n, side="right")) - 1
            W, c = self._weights(i.get("benchmark_id", ""))
            out[j] = W[:, b] @ L[:, j] + c[b]
        return sigmoid(out)
