"""Amortised logistic IRT with test-time MAP adaptation on (global) evidence.

    logit p(s, i) = beta_b + delta_{s,b} - d_i

fit (training benchmarks):
  1. Rasch fit per benchmark: pair ability theta_{s,b}, item difficulty d_i (items centred per benchmark).
  2. d_i ~ ridge(item features); prior var sigma_d^2 = cross-benchmark residual (cold-start realistic).
  3. beta_b = mean pair ability on b; prior N(mu_beta, sigma_beta^2) across training benchmarks.
  4. delta = theta - beta ~ ridge(subject features) blended with the subject's history on other
     benchmarks; prior var sigma_u^2 = cross-benchmark residual.
adapt (test time): per anonymous benchmark, block-coordinate Newton MAP of beta, delta (seen pairs)
  and d (seen items, if learn_items) under those priors, using every revealed label on the benchmark.
score: plug-in posterior means; with `marginalize`, the probit approximation over the summed
  posterior variances (wider uncertainty -> predictions pulled toward 0.5).
"""
import numpy as np

from ..features.item import ItemFeaturizer
from ..features.subject import SubjectFeaturizer, subject_identity, subject_key
from .base import Evidence, Predictor, item_key
from .simple import sigmoid


def ridge(X, y, lam, weights=None):
    X = np.asarray(X, float)
    sw = np.ones(len(y)) if weights is None else np.asarray(weights, float) / np.mean(weights)
    xm, ym = (sw @ X) / sw.sum(), float(sw @ y / sw.sum())
    Xc = X - xm
    w = np.linalg.solve(Xc.T @ (Xc * sw[:, None]) + lam * np.eye(X.shape[1]), Xc.T @ (sw * (y - ym)))
    return w, float(ym - xm @ w)


def rasch(P, J, ysum, n, nP, nJ, var, iters):
    theta, d = np.zeros(nP), np.zeros(nJ)
    for _ in range(iters):
        mu = sigmoid(theta[P] - d[J])
        theta += (np.bincount(P, ysum - n * mu, nP) - theta / var) / (np.bincount(P, n * mu * (1 - mu), nP) + 1 / var)
        mu = sigmoid(theta[P] - d[J])
        d += (-np.bincount(J, ysum - n * mu, nJ) - d / var) / (np.bincount(J, n * mu * (1 - mu), nJ) + 1 / var)
    return theta, d


class IRT(Predictor):
    name = "irt"
    # test-time-only params: excluded from the fit-cache key and re-applied after loading (fitcache.py)
    FIT_HELPERS = ("_fold_of",)   # called by fit; hashed into the fit-cache key (fitcache.py)
    RUNTIME_PARAMS = ("var_scale", "marginalize", "learn_bench", "learn_items", "adapt_iters", "clip", "subject_sharing",
                      "d_var_mult", "u_var_mult", "marg_scale")
    item_fields = ("text", "length", "kv")
    item_dim = 1024
    max_chars = 4000
    subject_fields = ("provider", "name", "think", "size", "date", "effort", "harness")
    subject_dim = 256
    ridge_item = 10.0
    ridge_subject = 10.0
    use_item_features = True
    use_subject_features = True
    use_history = True
    history_strength = 1.0
    learn_bench = True
    learn_items = True
    marginalize = True
    fit_iters = 40
    adapt_iters = 15
    fit_prior_var = 9.0
    var_floor = 0.05
    var_scale = 1.0          # multiplies all prior variances (calibration knob)
    beta_var_mult = 1.0      # multiplies only the benchmark-offset prior variance (cold-start hedge)
    d_var_mult = 1.0         # test-time multiplier of the item-difficulty prior variance (per-benchmark model)
    u_var_mult = 1.0         # test-time multiplier of the pair-ability prior variance (per-benchmark model)
    marg_scale = 1.0         # scales the variance used by the probit marginalisation in score
    # "centered": items predict within-benchmark difficulty; offset prior = mean of training offsets.
    # "absolute": items predict difficulty *including* the benchmark offset (text can carry "this kind
    #   of benchmark is easy"); offset prior N(0, cross-benchmark error of that prediction).
    item_target = "centered"
    balance_benchmarks = False   # weight items 1/n_items(benchmark) in the item ridge
    cv_groups = 10               # cross-benchmark residuals: benchmarks hashed into <= this many folds
    max_ridge_items = 200_000    # item-ridge fit on a seeded sample (memory); Rasch uses all items
    # subject_sharing: logit = beta_b + g_s + delta_{s,b} - d_i with a *global* subject ability g_s (keyed by
    #   model identity, shared across benchmarks) and a pair residual delta ~ N(0, var_pair). Adaptation is
    #   joint over all benchmarks, so a subject's labels on other test benchmarks inform this one.
    #   False: per-benchmark adaptation with delta_{s,b} ~ N(u(s), var_u) (the original model).
    subject_sharing = False

    # ---------------- fit ----------------
    def fit(self, train):
        self.ifeat = ItemFeaturizer(self.item_fields, self.item_dim, self.max_chars)
        self.sfeat = SubjectFeaturizer(self.subject_fields, self.subject_dim)
        nb, ni = len(train.bench_names), len(train.items)
        pk, P_resp = np.unique(train.resp_s * nb + train.resp_b, return_inverse=True)
        cells, inv = np.unique(P_resp * ni + train.resp_i, return_inverse=True)
        ysum, n = np.bincount(inv, train.resp_y), np.bincount(inv).astype(float)
        P, I = cells // ni, cells % ni
        items_used, J = np.unique(I, return_inverse=True)
        theta, d = rasch(P, J, ysum, n, len(pk), len(items_used), self.fit_prior_var, self.fit_iters)

        pair_b, pair_s = pk % nb, pk // nb
        item_b = train.item_bench[items_used]
        for b in np.unique(item_b):
            c = d[item_b == b].mean()
            d[item_b == b] -= c
            theta[pair_b == b] -= c

        # benchmark offset prior
        benches = np.unique(pair_b)
        beta = np.array([theta[pair_b == b].mean() for b in benches])
        self.mu_beta = float(beta.mean())
        self.var_beta = max(float(beta.var()) if len(beta) > 1 else 1.0, 0.25)

        fitdata = {"pk": pk, "P": P, "J": J, "ysum": ysum, "n": n, "items_used": items_used, "theta": theta.copy(),
                   "d": d.copy(), "pair_s": pair_s, "pair_b": pair_b}

        # item difficulty prior (ridge on a bounded, seeded sample of items)
        if len(items_used) > self.max_ridge_items:
            sel = np.sort(np.random.default_rng(0).choice(len(items_used), self.max_ridge_items, replace=False))
            items_used, d, item_b = items_used[sel], d[sel], item_b[sel]
        X = self.ifeat.transform([train.items[i] for i in items_used])
        wts = None
        if self.balance_benchmarks:
            cnt = np.bincount(item_b)
            wts = 1.0 / cnt[item_b]
        if self.item_target == "absolute":
            target = d - beta[np.searchsorted(benches, item_b)]   # logit(avg subject) = -target
            self.w_i, self.b_i, self.var_d, var_off = self._fit_prior(X, target, item_b, self.ridge_item,
                                                                       self.use_item_features, wts, split_offset=True)
            self.mu_beta, self.var_beta = 0.0, max(var_off, self.var_floor)
        else:
            self.w_i, self.b_i, self.var_d = self._fit_prior(X, d, item_b, self.ridge_item, self.use_item_features, wts)
        self.var_beta *= self.beta_var_mult

        # subject residual prior: features + history
        r = theta - beta[np.searchsorted(benches, pair_b)]
        Z = self.sfeat.transform([train.subjects[s] for s in pair_s])
        ident = [subject_identity(train.subjects[s]) for s in pair_s]
        resid = []
        pair_g = self._fold_of(pair_b)
        for g in np.unique(pair_g) if len(benches) > 1 else []:
            out, inn = pair_g == g, pair_g != g
            pred = self._fit_subject(Z[inn], r[inn], [ident[k] for k in np.nonzero(inn)[0]])
            resid.append(r[out] - np.array([pred(Z[k], ident[k]) for k in np.nonzero(out)[0]]))
        full = self._fit_subject(Z, r, ident, keep=True)
        if resid:
            self.var_u = max(float(np.mean(np.concatenate(resid) ** 2)), self.var_floor)
        else:
            self.var_u = max(float(np.mean((r - np.array([full(Z[k], ident[k]) for k in range(len(r))])) ** 2)) * 1.5, self.var_floor)
        # split pair-ability variance into global-subject and benchmark-specific parts
        acc = {}
        for k, v in zip(ident, r):
            acc.setdefault(k, []).append(v)
        dev = [(np.asarray(v) - np.mean(v), len(v)) for v in acc.values() if len(v) > 1]
        if dev:
            ss = sum(float((x ** 2).sum()) for x, _ in dev)
            dof = sum(c - 1 for _, c in dev)
            self.var_pair = max(ss / dof, self.var_floor)
        else:
            self.var_pair = self.var_u / 2
        self.var_g = max(self.var_u - self.var_pair, self.var_floor)
        self._fit_extra(train, fitdata)
        return self

    # ---------------- extension hooks (used by MIRT; no-ops here) ----------------
    def _fit_extra(self, train, fitdata):
        """Called at the end of fit with the full Rasch solution (cells P, J, ysum, n, theta, d, ...)."""

    def _bench_ctx(self, subj, items, P, J, warm):
        return None

    def _bench_step(self, ctx, base_eta, y):
        """Update extra per-benchmark parameters given the main-effect logits; return extra logits."""
        return 0.0

    def _bench_out(self, ctx, pidx):
        return {}

    def _score_extra(self, st, subject, item):
        """(extra logit mean, extra variance) for one input."""
        return 0.0, 0.0

    def _fold_of(self, bench):
        """Map benchmark indices to <= cv_groups folds (each benchmark stays whole)."""
        u = np.unique(bench)
        if len(u) <= self.cv_groups:
            return bench
        return (np.searchsorted(u, bench) % self.cv_groups)

    def _fit_prior(self, X, y, groups, lam, enabled, weights=None, split_offset=False):
        """Ridge prior + cross-group (leave-one-benchmark-out) residual variance.

        split_offset: also return the variance of the per-benchmark *mean* residual (how wrong the
        predicted benchmark offset is) and report the item variance net of it.
        """
        wsub = (lambda m: None) if weights is None else (lambda m: weights[m])
        if not enabled:
            w, b = np.zeros(X.shape[1]), float(np.mean(y))
        res, offs = [], []
        groups = self._fold_of(groups)
        for g in np.unique(groups) if len(np.unique(groups)) > 1 else []:
            m = groups != g
            wg, bg = ridge(X[m], y[m], lam, wsub(m)) if enabled else (w, float(np.mean(y[m])))
            r = y[~m] - (X[~m] @ wg + bg)
            offs.append(r.mean())
            res.append(r - r.mean() if split_offset else r)
        if enabled:
            w, b = ridge(X, y, lam, weights)
        var = np.mean(np.concatenate(res) ** 2) if res else np.mean((y - X @ w - b) ** 2) * 1.5
        out = (w, b, max(float(var), self.var_floor))
        if split_offset:
            out += (float(np.mean(np.square(offs))) if offs else 1.0,)
        return out

    def _fit_subject(self, Z, r, ident, keep=False):
        w, b = ridge(Z, r, self.ridge_subject) if self.use_subject_features else (np.zeros(Z.shape[1]), float(r.mean()))
        hist = {}
        if self.use_history:
            acc = {}
            for k, v in zip(ident, r):
                acc.setdefault(k, []).append(v)
            hist = {k: (float(np.mean(v)), len(v)) for k, v in acc.items()}
        if keep:
            self.w_s, self.b_s, self.hist = w, b, hist
        hs = self.history_strength

        def pred(z, idn):
            base = float(z @ w + b)
            h, c = hist.get(idn, (0.0, 0))
            return (c * h + hs * base) / (c + hs)
        return pred

    # ---------------- runtime priors (cached, pure) ----------------
    def _caches(self):
        if not hasattr(self, "_cache_u"):
            self._cache_u, self._cache_d = {}, {}
        return self._cache_u, self._cache_d

    def subject_prior(self, subject) -> float:
        cu, _ = self._caches()
        k = subject_key(subject)
        if k not in cu:
            base = float(self.sfeat.transform_one(subject) @ self.w_s + self.b_s)
            h, c = self.hist.get(subject_identity(subject), (0.0, 0))
            cu[k] = (c * h + self.history_strength * base) / (c + self.history_strength)
        return cu[k]

    def item_priors(self, items) -> np.ndarray:
        _, cd = self._caches()
        keys = [item_key(i) for i in items]
        todo = [(k, i) for k, i in zip(keys, items) if k not in cd]
        if todo:
            vals = self.ifeat.transform([i for _, i in todo]) @ self.w_i + self.b_i
            cd.update({k: float(v) for (k, _), v in zip(todo, vals)})
        return np.array([cd[k] for k in keys], float)

    # ---------------- adapt / score ----------------
    def adapt(self, labeled):
        state = {"ev": Evidence(), "fits": {}, "joint": None}
        return self.extend(state, labeled)

    def extend(self, state, new):
        touched = state["ev"].add(new)
        if self.subject_sharing:
            if touched:
                state["joint"] = self._fit_joint(state["ev"], state["joint"])
            return state
        for b in touched:
            state["fits"][b] = self._fit_bench(state["ev"].by_bench[b], state["fits"].get(b))
        return state

    def group_key(self, subject):
        """Key of the subject-level parameters (global ability g, MIRT skills)."""
        return subject_identity(subject) if self.subject_sharing else subject_key(subject)

    def _fit_joint(self, ev, warm=None):
        """MAP over all benchmarks at once: beta_b, g (per identity), delta (per pair), d (per item)."""
        vs = self.var_scale
        var_b, var_g, var_p, var_d = self.var_beta * vs, self.var_g * vs, self.var_pair * vs, self.var_d * vs
        bidx, gidx, pidx, iidx = {}, {}, {}, {}
        gsub, items, B, G, P, J, y = [], [], [], [], [], [], []
        for b, obs in ev.by_bench.items():
            bi = bidx.setdefault(b, len(bidx))
            for sk, ik, yy, s, i in obs:
                gk = self.group_key(s)
                if gk not in gidx:
                    gidx[gk] = len(gsub)
                    gsub.append(s)
                if ik not in iidx:
                    iidx[ik] = len(items)
                    items.append(i)
                B.append(bi)
                G.append(gidx[gk])
                P.append(pidx.setdefault((b, sk), len(pidx)))
                J.append(iidx[ik])
                y.append(yy)
        B, G, P, J, y = map(np.asarray, (B, G, P, J, y))
        nB, nG, nP, nJ = len(bidx), len(gsub), len(pidx), len(items)
        u = np.array([self.subject_prior(s) for s in gsub])
        yh = self.item_priors(items)
        beta, g, delta, d = np.full(nB, self.mu_beta), u.copy(), np.zeros(nP), yh.copy()
        if warm:
            for b, k in bidx.items():
                beta[k] = warm["beta"].get(b, (beta[k],))[0]
            for gk, k in gidx.items():
                g[k] = warm["g"].get(gk, (g[k],))[0]
            for pk, k in pidx.items():
                delta[k] = warm["delta"].get(pk, (0.0,))[0]
            for ik, k in iidx.items():
                d[k] = warm["d"].get(ik, (d[k],))[0]
        ctx = self._bench_ctx(gsub, items, G, J, warm)
        extra = 0.0

        def step(x, idx, n, prior_mean, prior_var, sign=1.0):
            mu = sigmoid(beta[B] + g[G] + delta[P] - d[J] + extra)
            x += (sign * np.bincount(idx, y - mu, n) - (x - prior_mean) / prior_var) / (np.bincount(idx, mu * (1 - mu), n) + 1 / prior_var)

        for _ in range(self.adapt_iters):
            if self.learn_bench:
                step(beta, B, nB, self.mu_beta, var_b)
            step(g, G, nG, u, var_g)
            step(delta, P, nP, 0.0, var_p)
            if self.learn_items:
                step(d, J, nJ, yh, var_d, sign=-1.0)
            if ctx is not None:
                extra = self._bench_step(ctx, beta[B] + g[G] + delta[P] - d[J], y)
        w = sigmoid(beta[B] + g[G] + delta[P] - d[J] + extra)
        w = w * (1 - w)
        post = lambda idx, n, pv: 1 / (np.bincount(idx, w, n) + 1 / pv)
        vb = post(B, nB, var_b) if self.learn_bench else np.full(nB, var_b)
        vg, vp = post(G, nG, var_g), post(P, nP, var_p)
        vd = post(J, nJ, var_d) if self.learn_items else np.full(nJ, var_d)
        out = {"beta": {b: (float(beta[k]), float(vb[k])) for b, k in bidx.items()},
               "g": {gk: (float(g[k]), float(vg[k])) for gk, k in gidx.items()},
               "delta": {pk: (float(delta[k]), float(vp[k])) for pk, k in pidx.items()},
               "d": {ik: (float(d[k]), float(vd[k])) for ik, k in iidx.items()}}
        if ctx is not None:
            out.update(self._bench_out(ctx, gidx))
        return out

    def _fit_bench(self, obs, warm=None):
        """MAP of beta, delta (seen pairs), d (seen items) for one benchmark's evidence."""
        vs = self.var_scale
        var_b, var_u, var_d = self.var_beta * vs, self.var_u * vs * self.u_var_mult, self.var_d * vs * self.d_var_mult
        pidx, iidx, subj, items = {}, {}, [], []
        for sk, ik, _, s, i in obs:
            if sk not in pidx:
                pidx[sk] = len(subj)
                subj.append(s)
            if ik not in iidx:
                iidx[ik] = len(items)
                items.append(i)
        P = np.array([pidx[o[0]] for o in obs])
        J = np.array([iidx[o[1]] for o in obs])
        y = np.array([o[2] for o in obs])
        u = np.array([self.subject_prior(s) for s in subj])
        yh = self.item_priors(items)
        nP, nJ = len(u), len(yh)
        beta, delta, d = self.mu_beta, u.copy(), yh.copy()
        ctx = self._bench_ctx(subj, items, P, J, warm)
        extra = 0.0
        if warm:  # warm start from the previous (smaller) evidence's solution
            beta = warm["beta"][0]
            for sk, k in pidx.items():
                if sk in warm["delta"]:
                    delta[k] = warm["delta"][sk][0]
            for ik, k in iidx.items():
                if ik in warm["d"]:
                    d[k] = warm["d"][ik][0]
        for _ in range(self.adapt_iters):
            if self.learn_bench:
                mu = sigmoid(beta + delta[P] - d[J] + extra)
                beta += ((y - mu).sum() - (beta - self.mu_beta) / var_b) / ((mu * (1 - mu)).sum() + 1 / var_b)
            mu = sigmoid(beta + delta[P] - d[J] + extra)
            delta += (np.bincount(P, y - mu, nP) - (delta - u) / var_u) / (np.bincount(P, mu * (1 - mu), nP) + 1 / var_u)
            if self.learn_items:
                mu = sigmoid(beta + delta[P] - d[J] + extra)
                d += (-np.bincount(J, y - mu, nJ) - (d - yh) / var_d) / (np.bincount(J, mu * (1 - mu), nJ) + 1 / var_d)
            if ctx is not None:
                extra = self._bench_step(ctx, beta + delta[P] - d[J], y)
        w = sigmoid(beta + delta[P] - d[J] + extra)
        w = w * (1 - w)
        vb = 1 / (w.sum() + 1 / var_b) if self.learn_bench else var_b
        vdel = 1 / (np.bincount(P, w, nP) + 1 / var_u)
        vd = 1 / (np.bincount(J, w, nJ) + 1 / var_d) if self.learn_items else np.full(nJ, var_d)
        return {"beta": (float(beta), float(vb)),
                "delta": {sk: (float(delta[k]), float(vdel[k])) for sk, k in pidx.items()},
                "d": {ik: (float(d[k]), float(vd[k])) for ik, k in iidx.items()},
                **(self._bench_out(ctx, pidx) if ctx is not None else {})}

    def score_batch(self, state, inputs):
        if self.subject_sharing:
            return self._score_joint(state, inputs)
        vs = self.var_scale
        yh = self.item_priors([i for _, i in inputs])
        m, v = np.empty(len(inputs)), np.empty(len(inputs))
        fits = state["fits"]
        for k, (s, i) in enumerate(inputs):
            st = fits.get(i.get("benchmark_id", ""))
            bm, bv = st["beta"] if st else (self.mu_beta, self.var_beta * vs)
            dm, dv = st["delta"].get(subject_key(s), (None, None)) if st else (None, None)
            if dm is None:
                dm, dv = self.subject_prior(s), self.var_u * vs * self.u_var_mult
            im, iv = st["d"].get(item_key(i), (None, None)) if st else (None, None)
            if im is None:
                im, iv = yh[k], self.var_d * vs * self.d_var_mult
            em, evar = self._score_extra(st, s, i)
            m[k], v[k] = bm + dm - im + em, bv + dv + iv + evar
        return sigmoid(m / np.sqrt(1 + np.pi * self.marg_scale * v / 8)) if self.marginalize else sigmoid(m)

    def _score_joint(self, state, inputs):
        vs = self.var_scale
        yh = self.item_priors([i for _, i in inputs])
        jt = state["joint"] or {"beta": {}, "g": {}, "delta": {}, "d": {}}
        m, v = np.empty(len(inputs)), np.empty(len(inputs))
        for k, (s, i) in enumerate(inputs):
            b = i.get("benchmark_id", "")
            bm, bv = jt["beta"].get(b, (self.mu_beta, self.var_beta * vs))
            gm, gv = jt["g"].get(self.group_key(s), (None, None))
            if gm is None:
                gm, gv = self.subject_prior(s), self.var_g * vs
            pm, pv = jt["delta"].get((b, subject_key(s)), (0.0, self.var_pair * vs))
            im, iv = jt["d"].get(item_key(i), (yh[k], self.var_d * vs))
            em, evar = self._score_extra(jt, s, i)
            m[k], v[k] = bm + gm + pm - im + em, bv + gv + pv + iv + evar
        return sigmoid(m / np.sqrt(1 + np.pi * v / 8)) if self.marginalize else sigmoid(m)

    def score(self, state, subject, item):
        return float(self.score_batch(state, [(subject, item)])[0])

    # ---------------- posterior components (for model-aware acquisition) ----------------
    def parts_batch(self, state, inputs):
        """Posterior parts per input: logit mean `mean` and the variances of its components
        `var_bench` (offset), `var_subject` (pair ability, incl. global ability), `var_item` (difficulty),
        `var_extra` (anything else). Same lookups as score_batch; used by acquisition.VOI."""
        vs = self.var_scale
        yh = self.item_priors([i for _, i in inputs])
        n = len(inputs)
        out = {k: np.empty(n) for k in ("mean", "var_bench", "var_subject", "var_item", "var_extra", "n_subjects")}
        jt = (state["joint"] or {"beta": {}, "g": {}, "delta": {}, "d": {}}) if self.subject_sharing else None
        for k, (s, i) in enumerate(inputs):
            b = i.get("benchmark_id", "")
            if jt is not None:
                bm, bv = jt["beta"].get(b, (self.mu_beta, self.var_beta * vs))
                gm, gv = jt["g"].get(self.group_key(s), (None, None))
                if gm is None:
                    gm, gv = self.subject_prior(s), self.var_g * vs
                pm, pv = jt["delta"].get((b, subject_key(s)), (0.0, self.var_pair * vs))
                dm, dv = gm + pm, gv + pv
                im, iv = jt["d"].get(item_key(i), (yh[k], self.var_d * vs))
                st = jt
            else:
                st = state["fits"].get(b)
                bm, bv = st["beta"] if st else (self.mu_beta, self.var_beta * vs)
                dm, dv = st["delta"].get(subject_key(s), (None, None)) if st else (None, None)
                if dm is None:
                    dm, dv = self.subject_prior(s), self.var_u * vs
                im, iv = st["d"].get(item_key(i), (None, None)) if st else (None, None)
                if im is None:
                    im, iv = yh[k], self.var_d * vs
            em, evar = self._score_extra(st, s, i)
            out["mean"][k], out["var_bench"][k], out["var_subject"][k] = bm + dm - im + em, bv, dv
            out["var_item"][k], out["var_extra"][k] = iv, evar
            out["n_subjects"][k] = state["ev"].n_pairs.get(b, 0)
        return out

    def prior_item_var(self) -> float:
        """Prior variance of an unlabeled item's difficulty (reference for acquisition)."""
        return self.var_d * self.var_scale
