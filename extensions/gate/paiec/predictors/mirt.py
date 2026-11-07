"""Multidimensional extension of IRT: content-based item loadings x subject skill vectors.

    logit p(s, i) = beta_b + delta_{s,b} - d_i + theta_s . (a_i - a_bar),   a_i = V^T x_i

- x_i are the same item content features as the IRT difficulty prior; V (F x K) is learned, so every
  new item gets a loading vector from its text alone (CAIMIRA-style amortisation).
- theta_s in R^K is a skill profile per model identity (shared across benchmarks). Prior for a new
  subject: ridge(subject features) blended with the identity's training profile (history_strength).
- a_bar centres loadings over training items, so the interaction has ~zero mean per subject and the
  main effects (beta, delta, d) keep their meaning and priors from IRT.fit.

fit: after IRT.fit, theta and V are fitted by Adam on the Rasch-residual structure (main-effect logits
  from the Rasch solution held fixed) on a seeded sample of <= mirt_items items, with L2 penalties.
adapt: inside IRT's per-benchmark Newton loop, each pair's theta gets a K x K Newton step toward its
  labels (prior N(theta0, var_theta I)); posterior variances feed the probit marginalisation.
"""
import numpy as np

from ..features.subject import subject_identity, subject_key
from .base import item_key
from .irt import IRT, ridge
from .simple import sigmoid


class MIRT(IRT):
    name = "mirt"
    ndim = 4
    mirt_items = 100_000
    mirt_steps = 300
    mirt_lr = 0.05
    lam_V = 1e-5          # L2 on the loading map (per mean response)
    lam_theta = 1e-5      # L2 on identity skill vectors
    init_scale = 0.5      # initial sd of loadings and skills: must be O(1), the bilinear form has a saddle at 0
    ridge_theta = 10.0
    adapt_theta = True
    theta_var_floor = 0.01

    # ---------------- fit ----------------
    def _fit_extra(self, train, fd):
        K = int(self.ndim)
        P, J, ysum, n = fd["P"], fd["J"], fd["ysum"], fd["n"]
        items_used, pair_s = fd["items_used"], fd["pair_s"]
        rng = np.random.default_rng(1)
        base = fd["theta"][P] - fd["d"][J]          # Rasch main-effect logits, held fixed
        if len(items_used) > self.mirt_items:
            keep = np.sort(rng.choice(len(items_used), self.mirt_items, replace=False))
            remap = -np.ones(len(items_used), np.int64)
            remap[keep] = np.arange(len(keep))
            m = remap[J] >= 0
            P, J, ysum, n, base = P[m], remap[J[m]], ysum[m], n[m], base[m]
            items = items_used[keep]
        else:
            items = items_used
        X = self.ifeat.transform([train.items[i] for i in items]).astype(np.float64)
        xbar = X.mean(0)
        idents = [subject_identity(train.subjects[s]) for s in pair_s]
        gnames, G_pair = np.unique(idents, return_inverse=True)
        G = G_pair[P]
        nG, F, N = len(gnames), X.shape[1], n.sum()

        xnorm = float(np.sqrt(np.mean(((X - xbar) ** 2).sum(1)))) or 1.0
        V = rng.normal(0, self.init_scale / xnorm, (F, K))      # -> loadings with sd ~ init_scale
        Th = rng.normal(0, self.init_scale, (nG, K))
        mV, vV, mT, vT = (np.zeros_like(V), np.zeros_like(V), np.zeros_like(Th), np.zeros_like(Th))
        b1, b2, eps = 0.9, 0.999, 1e-8
        for t in range(1, self.mirt_steps + 1):
            A = X @ V - xbar @ V
            TG, AJ = Th[G], A[J]
            eta = base + np.einsum("ck,ck->c", TG, AJ)
            g_eta = -(ysum - n * sigmoid(eta)) / N            # d(mean NLL)/d eta
            gA = np.stack([np.bincount(J, g_eta * TG[:, k], len(items)) for k in range(K)], 1)
            gV = X.T @ gA - np.outer(xbar, gA.sum(0)) + 2 * self.lam_V * V
            gT = np.stack([np.bincount(G, g_eta * AJ[:, k], nG) for k in range(K)], 1) + 2 * self.lam_theta * Th
            for p_, g_, m_, v_ in ((V, gV, mV, vV), (Th, gT, mT, vT)):
                m_ *= b1
                m_ += (1 - b1) * g_
                v_ *= b2
                v_ += (1 - b2) * g_ * g_
                p_ -= self.mirt_lr * (m_ / (1 - b1 ** t)) / (np.sqrt(v_ / (1 - b2 ** t)) + eps)
        self.V = V.astype(np.float32)
        self.a_bar = (xbar @ V).astype(np.float32)

        # skill prior: ridge from identity subject features, blended with the identity's own profile
        first = {}
        for k, s in zip(G_pair, pair_s):
            first.setdefault(k, s)
        Z = self.sfeat.transform([train.subjects[first[g]] for g in range(nG)])
        cols = [ridge(Z, Th[:, k], self.ridge_theta) for k in range(K)]
        self.W_th = np.stack([c[0] for c in cols], 1).astype(np.float32)
        self.b_th = np.array([c[1] for c in cols], np.float32)
        resid = Th - (Z @ self.W_th + self.b_th)
        self.var_theta = max(float(np.mean(resid ** 2)) * 1.5, self.theta_var_floor)
        counts = np.bincount(G_pair, minlength=nG)
        self.theta_table = {str(g): (Th[k].astype(np.float32), int(counts[k])) for k, g in enumerate(gnames)}

    # ---------------- runtime ----------------
    def item_priors(self, items):
        """Also caches each item's centred loading vector (one featurisation for both)."""
        _, cd = self._caches()
        if not hasattr(self, "_cache_a"):
            self._cache_a = {}
        ca = self._cache_a
        keys = [item_key(i) for i in items]
        todo = [(k, i) for k, i in zip(keys, items) if k not in cd or k not in ca]
        if todo:
            X = self.ifeat.transform([i for _, i in todo])
            vals = X @ self.w_i + self.b_i
            A = X @ self.V - self.a_bar
            for (k, _), v, a in zip(todo, vals, A):
                cd[k], ca[k] = float(v), a.astype(np.float64)
        return np.array([cd[k] for k in keys], float)

    def loadings(self, items) -> np.ndarray:
        self.item_priors(items)
        return np.stack([self._cache_a[item_key(i)] for i in items]) if items else np.zeros((0, self.ndim))

    def theta_prior(self, subject):
        if not hasattr(self, "_cache_th"):
            self._cache_th = {}
        k = subject_key(subject)   # not group_key: under sharing it is the identity, but the prior uses per-subject fields
        if k not in self._cache_th:
            base = self.sfeat.transform_one(subject) @ self.W_th + self.b_th
            tab, c = self.theta_table.get(subject_identity(subject), (None, 0))
            hs = self.history_strength
            mean = (c * tab + hs * base) / (c + hs) if c else base
            self._cache_th[k] = (mean.astype(np.float64), self.var_theta * hs / (c + hs) if c else self.var_theta)
        return self._cache_th[k]

    def _bench_ctx(self, subj, items, P, J, warm):
        pri = [self.theta_prior(s) for s in subj]
        th0 = np.stack([p[0] for p in pri])
        vth = np.array([p[1] for p in pri]) * self.var_scale
        th = th0.copy()
        keys = [self.group_key(s) for s in subj]
        if warm and "theta" in warm:
            for k, sk in enumerate(keys):
                if sk in warm["theta"]:
                    th[k] = warm["theta"][sk][0]
        A = self.loadings(items)
        return {"P": P, "J": J, "A": A, "th0": th0, "vth": vth, "th": th, "keys": keys,
                "H": None}

    def _bench_step(self, ctx, base_eta, y):
        P, AJ, th = ctx["P"], ctx["A"][ctx["J"]], ctx["th"]
        if self.adapt_theta:
            mu = sigmoid(base_eta + np.einsum("ok,ok->o", th[P], AJ))
            nP, K = th.shape
            g = np.zeros((nP, K))
            np.add.at(g, P, (y - mu)[:, None] * AJ)
            g -= (th - ctx["th0"]) / ctx["vth"][:, None]
            H = np.zeros((nP, K, K))
            np.add.at(H, P, (mu * (1 - mu))[:, None, None] * AJ[:, :, None] * AJ[:, None, :])
            H += np.eye(K)[None] / ctx["vth"][:, None, None]
            th += np.linalg.solve(H, g[..., None])[..., 0]
            ctx["H"] = H
        return np.einsum("ok,ok->o", th[P], AJ)

    def _bench_out(self, ctx, pidx):
        th, H = ctx["th"], ctx["H"]
        out = {}
        for k, sk in enumerate(ctx["keys"]):
            var = np.diag(np.linalg.inv(H[k])) if H is not None else np.full(th.shape[1], ctx["vth"][k])
            out[sk] = (th[k].copy(), var)
        return {"theta": out}

    def _score_extra(self, st, subject, item):
        a = self.loadings([item])[0]
        t = st.get("theta", {}).get(self.group_key(subject)) if st else None
        if t is None:
            mean, var = self.theta_prior(subject)
            var = np.full(len(a), var * self.var_scale)
        else:
            mean, var = t
        return float(mean @ a), float((a * a) @ var)
