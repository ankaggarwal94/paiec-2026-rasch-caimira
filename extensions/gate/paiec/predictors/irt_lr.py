"""IRT with a per-benchmark (test-time learned) regression of pair ability on subject descriptors.

    logit p(s, i) = beta_b + delta_{s,b} - d_i,   delta_{s,b} = x_s . w_b + r_{s,b},  r ~ N(0, var_r)
    w_b ~ N(wbar, S)

x_s (centred over training pairs) stacks the champion subject prior u(s) (feature ridge + identity history),
the identity's standardised history hz(s) (mean of delta / sd_benchmark over training benchmarks, shrunk), and
the dense subject descriptors (think flag, log size, release date, reasoning effort, with missing flags).

Why: which subjects are strong differs a lot by benchmark (e.g. reasoning vs not, size, recency). A single
global prior u(s) explains ~5% of held-out pair-ability variance; a per-benchmark loading vector w_b learned
from the ~20 subjects' labels on the new benchmark (global evidence) transfers "on this benchmark, size /
reasoning / general ability matter this much" to every pair, including pairs with few labels.

fit: after IRT.fit, per training benchmark ridge of delta (clipped) on x (out-of-fold u, leave-own-benchmark-out
  hz); wbar, S = mean and covariance of those w_c; var_r = pooled residual variance.
adapt: inside the per-benchmark Newton loop, w_b gets a closed-form Gaussian update from the pairs' working
  responses (z_s = delta_s + g_s / h_s with variance var_r + 1/h_s), and delta's prior mean is x_s . w_b.
"""
import numpy as np

from ..features.subject import subject_identity, subject_key
from .base import item_key
from .irt import IRT
from .simple import sigmoid

LR_FEATS = ("u", "hz", "think", "size", "size_na", "date", "date_na", "effort", "effort_na")


class IRTLR(IRT):
    name = "irt_lr"
    FIT_HELPERS = ("_stack",)   # builds the training design matrix behind lr_xbar / lr_wbar / lr_S
    RUNTIME_PARAMS = IRT.RUNTIME_PARAMS + ("lr_wvar_mult", "lr_rvar_mult", "lr_adapt_w")
    FIT_VERSION = 2
    lr_feats = LR_FEATS
    lr_clip = 4.0            # clip training delta (Rasch extremes of all-wrong/all-right pairs)
    lr_min_pairs = 15        # training benchmarks with fewer eligible pairs are not used for (wbar, S)
    lr_min_items = 20        # training pairs with fewer distinct items are not used
    lr_ridge = 1.0
    lr_hz_shrink = 3.0
    lr_wvar_mult = 1.0       # runtime: multiplies S
    lr_rvar_mult = 1.0       # runtime: multiplies var_r
    lr_adapt_w = True        # runtime: learn w_b at test time (False: fixed wbar)

    # ---------------- fit ----------------
    def _fit_extra(self, train, fd):
        theta, pair_s, pair_b = fd["theta"], fd["pair_s"], fd["pair_b"]
        nitems = np.bincount(fd["P"], minlength=len(theta))
        benches = np.unique(pair_b)
        bpos = np.searchsorted(benches, pair_b)
        beta = np.array([theta[pair_b == b].mean() for b in benches])
        sd = np.array([max(theta[pair_b == b].std(), 0.3) for b in benches])
        delta = theta - beta[bpos]
        ident = [subject_identity(train.subjects[s]) for s in pair_s]
        # out-of-fold champion prior u for training pairs
        Z = self.sfeat.transform([train.subjects[s] for s in pair_s])
        u_oof = np.zeros(len(theta))
        groups = self._fold_of(pair_b)
        for g in np.unique(groups):
            out, inn = np.nonzero(groups == g)[0], np.nonzero(groups != g)[0]
            pred = self._fit_subject(Z[inn], delta[inn], [ident[k] for k in inn])
            u_oof[out] = [pred(Z[k], ident[k]) for k in out]
        # standardised history, leave-own-benchmark-out for training rows
        z = np.clip(delta, -self.lr_clip, self.lr_clip) / sd[bpos]
        tab = {}
        for k, i in enumerate(ident):
            t = tab.setdefault(i, [0.0, 0])
            t[0] += z[k]
            t[1] += 1
        self.lr_hz_tab = {i: (float(v[0]), v[1]) for i, v in tab.items()}
        hz = np.array([(tab[i][0] - z[k]) / (tab[i][1] - 1 + self.lr_hz_shrink) for k, i in enumerate(ident)])
        D = Z[:, -7:].astype(np.float64)
        X = self._stack(u_oof, hz, D, [train.subjects[s].get("provider", "") for s in pair_s])
        ok = nitems >= self.lr_min_items
        self.lr_xbar = X[ok].mean(0)
        X = X - self.lr_xbar
        y = np.clip(delta, -self.lr_clip, self.lr_clip)
        W, rv, rn = [], [], []
        p = X.shape[1]
        for b in benches:
            m = np.nonzero(ok & (pair_b == b))[0]
            if len(m) < self.lr_min_pairs:
                continue
            A = np.c_[np.ones(len(m)), X[m]]
            reg = self.lr_ridge * np.eye(p + 1)
            reg[0, 0] = 1e-6
            w = np.linalg.solve(A.T @ A + reg, A.T @ y[m])
            r = y[m] - A @ w
            W.append(w[1:])
            rv.append(float(r @ r) / max(len(m) - p - 1, 1))
            rn.append(len(m))
        self.lr_nbench = len(W)
        if len(W) < 3:   # too little data (tests / tiny training sets): reduce to the IRT prior on u
            self.lr_wbar = np.array([1.0 if f == "u" else 0.0 for f in self.lr_feats])
            self.lr_S = 0.1 * np.eye(p)
            self.lr_var_r = self.var_u
            return
        W = np.array(W)
        self.lr_wbar = W.mean(0)
        self.lr_S = np.cov(W.T).reshape(p, p) + 1e-4 * np.eye(p)
        self.lr_var_r = max(float(np.average(rv, weights=rn)), self.var_floor)

    def _stack(self, u, hz, D, prov):
        """prov: provider strings; feature 'pv:<name>' is the one-hot of provider == name (lowercase)."""
        prov = [p.lower() for p in prov]
        cols = {f: np.array([float(p == f[3:]) for p in prov]) for f in self.lr_feats if f.startswith("pv:")}
        cols.update({"u": u, "hz": hz, "think": D[:, 0], "size": D[:, 1], "size_na": D[:, 2], "date": D[:, 3],
                "date_na": D[:, 4], "effort": D[:, 5], "effort_na": D[:, 6]})
        return np.stack([np.asarray(cols[f], np.float64) for f in self.lr_feats], 1)

    # ---------------- runtime ----------------
    def lr_x(self, subject):
        cu = self._lr_cache()
        k = subject_key(subject)
        if k not in cu:
            h, c = self.lr_hz_tab.get(subject_identity(subject), (0.0, 0))
            D = self.sfeat.transform_one(subject)[-7:].astype(np.float64)[None]
            x = self._stack(np.array([self.subject_prior(subject)]), np.array([h / (c + self.lr_hz_shrink)]), D,
                            [subject.get("provider", "")])[0]
            cu[k] = x - self.lr_xbar
        return cu[k]

    def _lr_cache(self):
        if not hasattr(self, "_cache_lrx"):
            self._cache_lrx = {}
        return self._cache_lrx

    def _fit_bench(self, obs, warm=None):
        vs = self.var_scale
        var_b, var_d = self.var_beta * vs, self.var_d * vs
        var_r = self.lr_var_r * self.lr_rvar_mult * vs
        S = self.lr_S * self.lr_wvar_mult * vs
        Si = np.linalg.inv(S)
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
        X = np.array([self.lr_x(s) for s in subj])
        yh = self.item_priors(items)
        nP, nJ = len(subj), len(yh)
        w = self.lr_wbar.copy()
        beta, d = self.mu_beta, yh.copy()
        if warm:
            beta, w = warm["beta"][0], warm["lr_w"].copy()
        delta = X @ w
        if warm:
            for sk, k in pidx.items():
                if sk in warm["delta"]:
                    delta[k] = warm["delta"][sk][0]
            for ik, k in iidx.items():
                if ik in warm["d"]:
                    d[k] = warm["d"][ik][0]
        Apost = Si
        for _ in range(self.adapt_iters):
            if self.learn_bench:
                mu = sigmoid(beta + delta[P] - d[J])
                beta += ((y - mu).sum() - (beta - self.mu_beta) / var_b) / ((mu * (1 - mu)).sum() + 1 / var_b)
            mu = sigmoid(beta + delta[P] - d[J])
            m0 = X @ w
            delta += (np.bincount(P, y - mu, nP) - (delta - m0) / var_r) / (np.bincount(P, mu * (1 - mu), nP) + 1 / var_r)
            if self.learn_items:
                mu = sigmoid(beta + delta[P] - d[J])
                d += (-np.bincount(J, y - mu, nJ) - (d - yh) / var_d) / (np.bincount(J, mu * (1 - mu), nJ) + 1 / var_d)
            if self.lr_adapt_w:
                mu = sigmoid(beta + delta[P] - d[J])
                h = np.bincount(P, mu * (1 - mu), nP) + 1e-9
                g = np.bincount(P, y - mu, nP)
                zs = delta + g / h
                wt = 1.0 / (var_r + 1.0 / h)
                Apost = X.T @ (X * wt[:, None]) + Si
                w = np.linalg.solve(Apost, X.T @ (wt * zs) + Si @ self.lr_wbar)
        Spost = np.linalg.inv(Apost)
        mu = sigmoid(beta + delta[P] - d[J])
        ww = mu * (1 - mu)
        h = np.bincount(P, ww, nP)
        vb = 1 / (ww.sum() + 1 / var_b) if self.learn_bench else var_b
        xs = np.einsum("ij,jk,ik->i", X, Spost, X)
        shrink = (1 / var_r) / (h + 1 / var_r)
        vdel = 1 / (h + 1 / var_r) + shrink ** 2 * xs
        vd = 1 / (np.bincount(J, ww, nJ) + 1 / var_d) if self.learn_items else np.full(nJ, var_d)
        return {"beta": (float(beta), float(vb)),
                "delta": {sk: (float(delta[k]), float(vdel[k])) for sk, k in pidx.items()},
                "d": {ik: (float(d[k]), float(vd[k])) for ik, k in iidx.items()},
                "lr_w": w, "lr_S": Spost}

    def score_batch(self, state, inputs):
        if self.subject_sharing:
            raise NotImplementedError("irt_lr supports per-benchmark adaptation only")
        vs = self.var_scale
        var_r = self.lr_var_r * self.lr_rvar_mult * vs
        S0 = self.lr_S * self.lr_wvar_mult * vs
        yh = self.item_priors([i for _, i in inputs])
        m, v = np.empty(len(inputs)), np.empty(len(inputs))
        fits = state["fits"]
        for k, (s, i) in enumerate(inputs):
            st = fits.get(i.get("benchmark_id", ""))
            bm, bv = st["beta"] if st else (self.mu_beta, self.var_beta * vs)
            dm, dv = st["delta"].get(subject_key(s), (None, None)) if st else (None, None)
            if dm is None:
                x = self.lr_x(s)
                w, S = (st["lr_w"], st["lr_S"]) if st else (self.lr_wbar, S0)
                dm, dv = float(x @ w), var_r + float(x @ S @ x)
            im, iv = st["d"].get(item_key(i), (None, None)) if st else (None, None)
            if im is None:
                im, iv = yh[k], self.var_d * vs
            m[k], v[k] = bm + dm - im, bv + dv + iv
        return sigmoid(m / np.sqrt(1 + np.pi * v / 8)) if self.marginalize else sigmoid(m)

