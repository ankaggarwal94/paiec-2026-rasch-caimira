"""IRT + a Brier-trained stacker learned from simulated replays on held-out *training* benchmarks.

fit:
  1. the base IRT on all training benchmarks (IRT.fit);
  2. inner replays: hold out `inner_benchmarks` training benchmarks (seeded), fit an IRT with the same
     params on the rest (`inner_folds` refits, each excluding its share), and replay the hosted
     protocol on the held-out ones: global evidence, <= `inner_max_subjects` subjects per benchmark,
     nested budgets 0..31, acquisition `inner_acq`. At every checkpoint, record per eval target the
     posterior components (beta, delta, d means/variances, priors), label counts and empirical rates;
  3. fit a logistic stacker on those features by minimising the ALC-weighted Brier score (each target
     weighs trials / n_responses(pair) * ALC weight of the budget), with a small ridge.
score: the base IRT's components -> stacker -> p. Pure numpy; budget is not visible at test time, so
  the features use the pair's label count instead.
"""
import hashlib
import os

import numpy as np

from ..features.subject import subject_key
from .base import item_key
from .irt import IRT
from .simple import sigmoid

ALC_W = {0: .1, 1: .2, 3: .2, 7: .2, 15: .2, 31: .1}
COLS = ("bm", "bv", "dm", "dv", "u", "im", "iv", "yh", "pk", "pn", "bk", "bn", "ik", "in_")


def _lg(p):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def design(c, kind):
    """Stacker design matrix from component columns c (dict of equal-length arrays)."""
    m = c["bm"] + c["dm"] - c["im"]
    v = c["bv"] + c["dv"] + c["iv"]
    z = m / np.sqrt(1 + np.pi * v / 8)
    one = np.ones_like(z)
    pn = c["pn"]
    bins = [pn == 0, pn == 1, (pn >= 2) & (pn <= 3), (pn >= 4) & (pn <= 7), (pn >= 8) & (pn <= 15), pn >= 16]
    bins = [b.astype(float) for b in bins]
    if kind == "platt":
        return np.column_stack([z, one])
    if kind == "platt_n":
        return np.column_stack([z * e for e in bins] + bins)
    lpn, lin, lbn = np.log1p(pn), np.log1p(c["in_"]), np.log1p(c["bn"])
    sv = np.sqrt(v)
    base = [z * e for e in bins] + bins + [m, sv, z * sv, z * lin, z * lbn, lin, lbn]
    if kind == "lin":
        return np.column_stack(base)
    du, di = c["dm"] - c["u"], c["im"] - c["yh"]
    comp = [c["bm"], c["u"], c["yh"], du, di, np.sqrt(c["bv"]), np.sqrt(c["dv"]), np.sqrt(c["iv"]),
            du * lpn, di * lin, c["bm"] * lbn]
    if kind == "comp":
        return np.column_stack(base + comp)
    pr, ir, br = (c["pk"] + 1) / (pn + 2), (c["ik"] + 1) / (c["in_"] + 2), (c["bk"] + 1) / (c["bn"] + 2)
    emp = [_lg(pr) * (pn > 0), _lg(ir) * (c["in_"] > 0), _lg(br) * (c["bn"] > 0), _lg(pr) * lpn, _lg(ir) * lin]
    if kind == "full":
        return np.column_stack(base + comp + emp)
    raise ValueError(f"unknown stack_kind {kind!r}")


def fit_brier(A, y, w, lam):
    """Logistic model minimising sum w (sigmoid(A b + c) - y)^2 + lam * W * |b|^2 (standardised columns,
    unpenalised intercept c)."""
    from scipy.optimize import minimize
    mu, sd = A.mean(0), A.std(0)
    sd[sd < 1e-12] = 1.0
    An = np.column_stack([(A - mu) / sd, np.ones(len(A))])
    W = w.sum()
    pen = np.ones(An.shape[1])
    pen[-1] = 0.0

    def f(b):
        p = sigmoid(An @ b)
        r = p - y
        return float(w @ r ** 2 + lam * W * (pen * b) @ b), An.T @ (w * 2 * r * p * (1 - p)) + 2 * lam * W * pen * b
    res = minimize(f, np.zeros(An.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": 2000})
    coef = res.x[:-1] / sd
    return coef, float(res.x[-1] - mu @ coef)


class StackedIRT(IRT):
    name = "stacked_irt"
    FIT_VERSION = 1
    RUNTIME_PARAMS = IRT.RUNTIME_PARAMS + ("stack_apply", "stack_kind")
    KINDS = ("platt", "platt_n", "lin", "comp", "full")
    stack_apply = True
    stack_kind = "full"         # which of the fitted stackers to apply (all KINDS are fitted)
    stack_lam = 1e-4
    inner_benchmarks = 30       # training benchmarks held out for inner replays (eligible ones only)
    inner_folds = 2             # IRT refits; each excludes its share of the inner benchmarks
    inner_seeds = (0,)
    inner_min_items = 80
    inner_max_subjects = 20
    inner_max_eval = 100
    inner_seen_from = ()        # optional: evaluate only subjects whose identity appears in these benchmarks
    inner_acq = None            # acquisition config for inner replays (default uncertainty r=0.15)
    stackers = None             # kind -> (coef, intercept)

    def fit(self, train):
        super().fit(train)
        rows = self._fit_inner(train)
        if rows is not None:
            c, y, w = rows
            dump = os.environ.get("PAIEC_META_DUMP")
            if dump:   # keep the inner-replay rows for offline analysis
                os.makedirs(dump, exist_ok=True)
                tag = hashlib.sha1(repr(sorted(set(train.resp_b.tolist()))).encode()).hexdigest()[:10]
                np.savez_compressed(os.path.join(dump, f"inner_{tag}.npz"), y=y, w=w, **c)
            self.stackers = {k: fit_brier(design(c, k), y, w, self.stack_lam) for k in self.KINDS}
            self.inner_rows = int(len(y))
        return self

    def _fit_inner(self, train):
        from ..acquisition import build_policy
        from ..features.subject import subject_identity
        from ..metrics import BUDGETS
        from ..sim.replay import build_pairs, replay
        names = [train.bench_names[b] for b in np.unique(train.resp_b)]
        elig = self._eligible(train, names)
        if len(elig) < 2:
            return None
        rng = np.random.default_rng(12345)
        held = list(rng.permutation(elig)[: self.inner_benchmarks])
        allowed = None
        if self.inner_seen_from:
            ref = np.isin(train.resp_b, [train.bench_names.index(b) for b in self.inner_seen_from if b in train.bench_names])
            seen = {subject_identity(train.subjects[s]) for s in np.unique(train.resp_s[ref])}
            allowed = {k for k, s in enumerate(train.subjects) if subject_identity(s) in seen}
        params = {k: getattr(self, k) for k in vars(self) if not k.startswith("_") and hasattr(IRT, k)}
        acq = dict(self.inner_acq or {"name": "uncertainty", "radius": 0.15})
        cols = {k: [] for k in COLS}
        ys, ws = [], []
        for f in range(self.inner_folds):
            grp = held[f::self.inner_folds]
            if not grp:
                continue
            rest = [b for b in names if b not in grp]
            inner = IRT(**params).fit(train.restrict(rest))
            for seed in self.inner_seeds:
                pairs = build_pairs(train, grp, 0, seed, self.inner_min_items, self.inner_max_subjects,
                                    self.inner_max_eval, allowed)
                if not pairs:
                    continue
                log = []
                orig = inner.predict_batch

                def pb(inputs, labeled=None, _orig=orig, _inner=inner, _log=log):
                    p = _orig(inputs, labeled)
                    _log.append(components(_inner, _inner._state(labeled), inputs))
                    return p
                inner.predict_batch = pb
                inner.reset_runtime()
                replay(inner, build_policy(acq), train, pairs, "global", True)
                del inner.predict_batch
                for bi, n in enumerate(BUDGETS):
                    for pi, pr in enumerate(pairs):
                        comp = log[bi * len(pairs) + pi]
                        nr = sum(len(t) for t in pr.evl_ys)
                        for k in COLS:
                            cols[k].append(comp[k])
                        ys.append([t.mean() for t in pr.evl_ys])
                        ws.append([ALC_W[n] * len(t) / nr for t in pr.evl_ys])
        if not ys:
            return None
        return {k: np.concatenate(v) for k, v in cols.items()}, np.concatenate(ys), np.concatenate(ws)

    def _eligible(self, train, names):
        """Training benchmarks with >= 2 subjects having >= inner_min_items distinct source items."""
        out = []
        b_of = train.resp_b
        for name in names:
            b = train.bench_names.index(name)
            m = b_of == b
            s, i = train.resp_s[m], train.resp_i[m]
            src = np.array([train.item_ids[k].split("|", 1)[0] for k in np.unique(i)])
            src_of = dict(zip(np.unique(i), src))
            n_ok = 0
            for ss in np.unique(s):
                if len({src_of[k] for k in np.unique(i[s == ss])}) >= self.inner_min_items:
                    n_ok += 1
                    if n_ok >= 2:
                        out.append(name)
                        break
        return out

    # ---------------- runtime ----------------
    def score_batch(self, state, inputs):
        if self.subject_sharing or not self.stack_apply or not self.stackers:
            return super().score_batch(state, inputs)
        coef, icpt = self.stackers[self.stack_kind]
        return sigmoid(design(components(self, state, inputs), self.stack_kind) @ coef + icpt)


def components(pred, state, inputs):
    """Posterior components + counts per input (non-sharing IRT state)."""
    vs = pred.var_scale
    ev, fits = state["ev"], state["fits"]
    yh = pred.item_priors([i for _, i in inputs])
    n = len(inputs)
    out = {k: np.empty(n) for k in COLS}
    icnt = state.setdefault("_icnt", {})
    for k, (s, i) in enumerate(inputs):
        b = i.get("benchmark_id", "")
        obs = ev.by_bench.get(b, [])
        ic = icnt.get(b)
        if ic is None or ic[0] != len(obs):   # per-item label counts, keyed by evidence size
            cnt = {}
            for _, ik, y, _, _ in obs:
                e = cnt.setdefault(ik, [0.0, 0])
                e[0] += y
                e[1] += 1
            ic = icnt[b] = (len(obs), cnt)
        st = fits.get(b)
        bm, bv = st["beta"] if st else (pred.mu_beta, pred.var_beta * vs)
        u = pred.subject_prior(s)
        dm, dv = st["delta"].get(subject_key(s), (None, None)) if st else (None, None)
        if dm is None:
            dm, dv = u, pred.var_u * vs * pred.u_var_mult
        ikey = item_key(i)
        im, iv = st["d"].get(ikey, (None, None)) if st else (None, None)
        if im is None:
            im, iv = yh[k], pred.var_d * vs * pred.d_var_mult
        pk, pn = ev.pair_counts(s, i)
        bk, bn = ev.bench_counts(i)
        ik_, in_ = ic[1].get(ikey, (0.0, 0))
        for name, val in zip(COLS, (bm, bv, dm, dv, u, im, iv, yh[k], pk, pn, bk, bn, ik_, in_)):
            out[name][k] = val
    return out
