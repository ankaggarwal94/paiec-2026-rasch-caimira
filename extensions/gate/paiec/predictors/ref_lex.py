"""Collaborator's `lex_cosine` submission (c14_lex_cosine_p4_a1), ported to the harness with identical arithmetic.

Stateless pair-local model: an unseen item of subject s on benchmark b is scored from the labels *s* already has on *b*,
as a smoothed mean weighted by token-set similarity to the query item:

    w_j = floor + sim(q, x_j) ** power,   est = sum w_j y_j / sum w_j,   p = (alpha*prior + n*est) / (alpha + n)

No training data, no cross-benchmark information, no other subjects' evidence; with no own labels p = prior.
Verified against the original `predict` in scripts/check_ref_lex.py. Note: the harness truncates stored item text at
`max_item_chars` (8000), the zip sees the full text.
"""
import json
import math
import re

import numpy as np

from ..features.subject import subject_key
from .base import Predictor

TOKEN = re.compile(r"\w+", flags=re.UNICODE)


class RefLex(Predictor):
    name = "ref_lex"
    RUNTIME_PARAMS = ("alpha", "prior", "power", "floor", "metric", "blend")
    alpha = 1.0
    prior = 0.5
    power = 4.0
    floor = 0.1
    metric = "cosine"     # cosine | jaccard
    blend = 1.0
    clip = (0.0001, 0.9999)

    def _tok(self, item):
        c = self.__dict__.setdefault("_cache_tok", {})
        k = (item.get("benchmark_id", ""), item.get("item_features", ""), item.get("interactors", ""),
             item.get("item_content", ""))
        t = c.get(k)
        if t is None:
            body = {key: v for key, v in item.items() if key != "benchmark_id"}
            t = c[k] = frozenset(TOKEN.findall(json.dumps(body, sort_keys=True, separators=(",", ":")).casefold()))
        return t

    def adapt(self, labeled):
        return self.extend({}, labeled)

    def extend(self, state, new):
        for (s, i), y in new:
            state.setdefault((i.get("benchmark_id", ""), subject_key(s)), []).append((self._tok(i), float(y)))
        return state

    def _sim(self, a, b):
        if not a or not b:
            return 0.0
        o = len(a & b)
        return o / len(a | b) if self.metric == "jaccard" else o / math.sqrt(len(a) * len(b))

    def score(self, state, subject, item):
        own = state.get((item.get("benchmark_id", ""), subject_key(subject)), [])
        n = len(own)
        total = math.fsum(y for _, y in own)
        base = (self.alpha * self.prior + total) / (self.alpha + n)
        result = base
        if n:
            q = self._tok(item)
            w = [self.floor + self._sim(q, t) ** self.power for t, _ in own]
            den = math.fsum(w)
            if den:
                est = math.fsum(wi * y for wi, (_, y) in zip(w, own)) / den
                local = (self.alpha * self.prior + n * est) / (self.alpha + n)
                result = (1.0 - self.blend) * base + self.blend * local
        return result

    def score_batch(self, state, inputs):
        return np.array([self.score(state, s, i) for s, i in inputs], float)
