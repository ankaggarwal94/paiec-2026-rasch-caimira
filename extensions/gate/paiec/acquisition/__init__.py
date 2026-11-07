"""Streaming acquisition policies: __call__(input, prediction, labeled, context) -> bool.

Hosted protocol (from the starting kit, via the collaborator's replay):
candidates of one pair's acquisition pool arrive in random order; each is accepted (label
revealed) or skipped forever. context = {subject_id, benchmark_id, labels_acquired,
labels_remaining (of 31), max_labels, items_remaining (incl. this candidate)}.
Every policy here is stateless and enforces the fill rule: accept when
items_remaining <= labels_remaining, reject when labels_remaining == 0.

Skip room per pair is |acquisition pool| - 31, so selective policies only matter on
larger pairs. Selective policies bias predictors that ignore item difficulty.
"""
import hashlib
import json

import numpy as np


class Policy:
    name = "base"

    def __init__(self, **params):
        for k, v in params.items():
            if not hasattr(type(self), k):
                raise ValueError(f"{type(self).__name__} has no parameter {k!r}")
            setattr(self, k, v)

    def __call__(self, input, prediction=None, labeled=None, context=None) -> bool:
        rem, items = context["labels_remaining"], context["items_remaining"]
        if rem <= 0:
            return False
        if items <= rem:
            return True
        return bool(self.decide(input, prediction, context, rem / items, labeled or []))

    def bind(self, predictor) -> "Policy":
        """Give the policy access to the fitted predictor (used by model-aware policies)."""
        self._predictor = predictor
        return self

    def decide(self, input, p, context, rate, labeled) -> bool:
        raise NotImplementedError


class AcceptAll(Policy):
    """Take the first 31 candidates. Uniform subset (order is random); fills budgets earliest."""
    name = "accept_all"

    def decide(self, input, p, context, rate, labeled):
        return True


class OfficialRandom(Policy):
    """Platform default: deterministic hash-uniform u < labels_remaining / items_remaining."""
    name = "official_random"

    def decide(self, input, p, context, rate, labeled):
        key = json.dumps([context, input], sort_keys=True, separators=(",", ":"))
        u = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2 ** 64
        return u < rate


class Uncertainty(Policy):
    """Accept when |p - 0.5| <= radius (the paiec_baseline BLE rule, plus fill guarantee)."""
    name = "uncertainty"
    radius = 0.15
    radius_first = None   # optional radius for the pair's first label (B1), when it differs

    def decide(self, input, p, context, rate, labeled):
        r = self.radius_first if self.radius_first is not None and context["labels_acquired"] == 0 else self.radius
        return abs(p - 0.5) <= r


class Information(Policy):
    """Accept when Fisher information 4p(1-p) >= (1 - rate)^gamma.

    rate = labels_remaining / items_remaining: when skip room is plentiful (rate small) only
    near-0.5 items pass; as room runs out the bar drops to 0. gamma=0 -> accept_all.
    """
    name = "information"
    gamma = 1.0

    def decide(self, input, p, context, rate, labeled):
        return 4 * p * (1 - p) >= (1 - rate) ** self.gamma


class Quantile(Policy):
    """Accept iff the candidate's information 4p(1-p) is in the top `rate` fraction of a reference set.

    The reference set is the same benchmark's distinct items already revealed for other subjects (global
    evidence), scored for *this* subject with the bound predictor. They come from the same benchmark's
    acquisition pools, so they stand in for the candidates still to come. Accepting the top
    labels_remaining / items_remaining fraction fills the budget at the right rate while keeping the
    most informative items. Falls back to Information(gamma) without a predictor or with < min_ref items.
    """
    name = "quantile"
    min_ref = 30
    max_ref = 400
    slack = 1.0          # >1 accepts more freely (less risk of running out of candidates)
    gamma = 1.0

    def decide(self, input, p, context, rate, labeled):
        pred = getattr(self, "_predictor", None)
        subject, item = input
        bid = item.get("benchmark_id", "")
        ref, seen = [], set()
        if pred is not None:
            for (s, i), _ in reversed(labeled):
                if i.get("benchmark_id", "") != bid:
                    continue
                k = (i.get("item_content", "")[:256], i.get("item_features", ""))
                if k not in seen:
                    seen.add(k)
                    ref.append((subject, i))
                    if len(ref) >= self.max_ref:
                        break
        info = 4 * p * (1 - p)
        if len(ref) < self.min_ref:
            return info >= (1 - rate) ** self.gamma
        pr = pred.predict_batch(ref, labeled)
        thr = np.quantile(4 * pr * (1 - pr), max(0.0, 1 - rate * self.slack))
        return info >= thr


def _parts(policy, input, labeled):
    pred = getattr(policy, "_predictor", None)
    fn = getattr(pred, "posterior_parts", None)
    return fn(input, labeled) if fn is not None else None


class VOI(Policy):
    """Value-of-information rule on the predictor's posterior components (Laplace / linearised IRT).

    A label on candidate (s, i) carries expected Fisher information w = E[σ'(η)] ≈ p(1-p)/κ with
    κ = √(1 + πV/8), V = v_bench + v_subject + v_item (+ v_extra, counted with the subject). In the
    linearised model its expected reduction of the posterior variance of a target logit c is
    w·Cov(η_i, c)² / (1 + wV). Targets, with diagonal-posterior covariances:
      - this pair's eval items:            Cov = v_bench + v_subject           (weight 1)
      - other subjects scoring item i:     Cov = v_bench + v_item              (weight item_weight)
      - other pairs on the benchmark:      Cov = v_bench                       (weight bench_weight)
    Efficiency = value / value of a reference candidate (logit mean 0, unlabeled item, i.e. prior
    item variance). Accept iff efficiency >= tau. Pinned items (low v_item) carry more pair
    information; unlabeled items carry more item information (item_weight). With item_weight = 0 and
    no predictor parts, this reduces to an uncertainty band (fallback_radius).
    info: "expected" (p(1-p)/κ), "marginal" (p(1-p) of the marginal p), "plugin" (σ'(mean)).
    """
    name = "voi"
    tau = 0.9
    item_weight = 0.0
    bench_weight = 0.0
    info = "expected"
    fallback_radius = 0.15
    # weights: "const" uses item_weight / bench_weight as given. "auto" scales them by the expected target
    #   counts: each other subject scores item i with prob. 1/2 (50/50 split) and this pair has about
    #   N = labels_acquired + items_remaining eval items, so item ×(S-1)/(2N) and bench ×(S-1), with S the
    #   number of subjects seen on the benchmark (incl. this one), at least min_subjects.
    weights = "const"
    min_subjects = 2

    def _value(self, m, vb, vs, vi, lam, mu):
        V = vb + vs + vi
        k = np.sqrt(1 + np.pi * V / 8)
        if self.info == "plugin":
            q = 1 / (1 + np.exp(-m))
            w = q * (1 - q)
        else:
            q = 1 / (1 + np.exp(-m / k))
            w = q * (1 - q) / (k if self.info == "expected" else 1.0)
        return w / (1 + w * V) * ((vb + vs) ** 2 + lam * (vb + vi) ** 2 + mu * vb ** 2)

    def decide(self, input, p, context, rate, labeled):
        pt = _parts(self, input, labeled)
        if pt is None:
            return abs(p - 0.5) <= self.fallback_radius
        vb, vs, vi = pt["var_bench"], pt["var_subject"] + pt["var_extra"], pt["var_item"]
        lam, mu = self.item_weight, self.bench_weight
        if self.weights == "auto":
            others = max(pt.get("n_subjects", 0) + (context["labels_acquired"] == 0), self.min_subjects) - 1
            lam *= others / (2.0 * (context["labels_acquired"] + context["items_remaining"]))
            mu *= others
        ref = self._value(0.0, vb, vs, self._predictor.prior_item_var(), lam, mu)
        return self._value(pt["mean"], vb, vs, vi, lam, mu) >= self.tau * ref


class Coverage(Policy):
    """Uncertainty band whose radius depends on whether the candidate item already has labels
    (from other subjects): radius_seen for labeled ("pinned") items, radius_new for unlabeled ones.
    Labeled = posterior item variance below prior_item_var * seen_frac."""
    name = "coverage"
    radius_seen = 0.15
    radius_new = 0.15
    seen_frac = 0.99

    def decide(self, input, p, context, rate, labeled):
        pt = _parts(self, input, labeled)
        seen = pt is not None and pt["var_item"] < self._predictor.prior_item_var() * self.seen_frac
        return abs(p - 0.5) <= (self.radius_seen if seen else self.radius_new)


POLICIES = {c.name: c for c in (AcceptAll, OfficialRandom, Uncertainty, Information, Quantile, VOI, Coverage)}


def build_policy(cfg: dict) -> Policy:
    """Drop parameters that belong to *other* policies (so `--set acquisition.name=...` can switch
    policies over a config that sets e.g. `radius`); unknown keys still raise."""
    cfg = dict(cfg)
    cls = POLICIES[cfg.pop("name")]
    others = {k for c in POLICIES.values() if c is not cls for k in vars(c) if not k.startswith("_")}
    return cls(**{k: v for k, v in cfg.items() if hasattr(cls, k) or k not in others})
