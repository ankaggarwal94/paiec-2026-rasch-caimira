"""IRT with test-time hierarchical adaptation of per-benchmark hyperparameters (direction `hier`).

    logit p(s, i) = beta_b + delta_{s,b} - d_i
    delta_{s,b} ~ N(lam_u,b * u(s), s2_u,b)            lam_u,b ~ N(1, tau_u^2),  s2_u,b ~ Inv-chi2(nu_u, var_u)
    d_i         ~ N(c_b + lam_d,b * (yh_i - c_b), s2_d,b)  lam_d,b ~ N(1, tau_d^2),  s2_d,b ~ Inv-chi2(nu_d, var_d)

c_b is the mean item prior over the benchmark's labeled items (a pure function of the evidence).
At test time each benchmark's (lam, s2) pairs are learned by Laplace-EM inside the Newton loop from
all revealed labels on that benchmark (global evidence), shrunk to the training values by the
hyperpriors (nu = pseudo-count, tau = prior sd of the slope). nu = 0 / tau = 0 switch a part off, so
with all four off this is exactly IRT.

Every new parameter is test-time only (RUNTIME_PARAMS), so the fitted object is IRT's fit.
Runtime methods are deliberately not named `_fit*` (the fit cache hashes those).
"""
import numpy as np

from ..features.subject import subject_key
from .base import item_key
from .irt import IRT
from .simple import sigmoid


class IRTH(IRT):
    name = "irt_h"
    nu_u = 0.0          # pseudo-count of the subject-variance hyperprior (0 = fixed global var_u)
    nu_d = 0.0          # same for the item variance
    tau_u = 0.0         # prior sd of the subject slope on u(s) (0 = slope fixed at 1)
    tau_d = 0.0         # prior sd of the item slope on the content prior (0 = fixed at 1)
    em_start = 0        # Newton iteration from which the hyperparameters are updated
    min_var_mult = 0.05  # learned variance >= this x the training variance
    max_var_mult = 20.0
    score_var_mult = 1.0  # multiplies the summed posterior variance in the probit marginalisation
    # vi: mean-field Gaussian VI instead of the joint MAP. Each Newton step uses the expected likelihood:
    #   residual y - E[sigmoid(eta)] ~= y - sigmoid(m / kappa), curvature E[sigmoid'] ~= sigmoid'(m / kappa) / kappa,
    #   kappa = sqrt(1 + pi v / 8), v = summed current posterior variances of the observation's terms.
    #   The joint MAP attenuates the offset/abilities when items carry large, weakly identified variances.
    vi = False
    vi_frac = 1.0   # scales the variance inside the VI expectation (0 = MAP, 1 = mean-field VI)
    # runtime multipliers of the individual prior variances (on top of var_scale), for adapt and score
    b_mult = 1.0
    u_mult = 1.0
    d_mult = 1.0
    RUNTIME_PARAMS = IRT.RUNTIME_PARAMS + ("nu_u", "nu_d", "tau_u", "tau_d", "em_start", "min_var_mult",
                                           "max_var_mult", "score_var_mult", "vi", "vi_frac", "b_mult", "u_mult", "d_mult")

    def _prior_vars(self):
        vs = self.var_scale
        return self.var_beta * vs * self.b_mult, self.var_u * vs * self.u_mult, self.var_d * vs * self.d_mult

    def extend(self, state, new):
        if self.subject_sharing:
            return super().extend(state, new)
        touched = state["ev"].add(new)
        for b in touched:
            state["fits"][b] = self._adapt_h(state["ev"].by_bench[b], state["fits"].get(b))
        return state

    @staticmethod
    def _hyper(m, v, x, s2_0, nu, tau, s2_prev, lo, hi):
        """MAP of (slope, variance) in m ~ N(lam x, s2) given posteriors N(m, v); priors lam ~ N(1, tau^2),
        s2 ~ scaled-Inv-chi2(nu, s2_0)."""
        lam = 1.0
        s2 = s2_prev
        for _ in range(3):
            if tau > 0:
                lam = (float(x @ m) / s2 + 1 / tau ** 2) / (float(x @ x) / s2 + 1 / tau ** 2)
            if nu > 0:
                ss = float(((m - lam * x) ** 2 + v).sum())
                s2 = min(max((nu * s2_0 + ss) / (nu + len(m)), lo * s2_0), hi * s2_0)
        return lam, s2

    def _adapt_h(self, obs, warm=None):
        var_b, var_u, var_d = self._prior_vars()
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
        cb = float(yh.mean())
        yc = yh - cb
        lam_u, s2u, lam_d, s2d = 1.0, var_u, 1.0, var_d
        beta, delta, d = self.mu_beta, u.copy(), yh.copy()
        if warm:
            beta = warm["beta"][0]
            for sk, k in pidx.items():
                if sk in warm["delta"]:
                    delta[k] = warm["delta"][sk][0]
            for ik, k in iidx.items():
                if ik in warm["d"]:
                    d[k] = warm["d"][ik][0]
            h = warm.get("hyp")
            if h:
                lam_u, s2u, lam_d, s2d = h["lam_u"], h["s2u"], h["lam_d"], h["s2d"]
        learn_u = self.nu_u > 0 or self.tau_u > 0
        learn_d = self.nu_d > 0 or self.tau_d > 0
        lo, hi = self.min_var_mult, self.max_var_mult
        vb, vdel, vd = var_b, np.full(nP, s2u), np.full(nJ, s2d)
        if warm and self.vi and "vb" in warm:
            vb = warm["vb"]
            vdel = np.array([warm["delta"].get(sk, (0, s2u))[1] for sk in pidx])
            vd = np.array([warm["d"].get(ik, (0, s2d))[1] for ik in iidx])

        def lik():
            eta = beta + delta[P] - d[J]
            if not self.vi:
                mu = sigmoid(eta)
                return mu, mu * (1 - mu)
            kap = np.sqrt(1 + np.pi * self.vi_frac * ((vb if self.learn_bench else 0.0) + vdel[P] + vd[J]) / 8)
            mu = sigmoid(eta / kap)
            return mu, mu * (1 - mu) / kap

        for it in range(self.adapt_iters):
            if self.learn_bench:
                mu, w = lik()
                hb = w.sum() + 1 / var_b
                beta += ((y - mu).sum() - (beta - self.mu_beta) / var_b) / hb
                vb = 1 / hb
            mu, w = lik()
            hP = np.bincount(P, w, nP) + 1 / s2u
            delta += (np.bincount(P, y - mu, nP) - (delta - lam_u * u) / s2u) / hP
            vdel = 1 / hP
            if learn_u and it >= self.em_start:
                lam_u, s2u = self._hyper(delta, vdel, u, var_u, self.nu_u, self.tau_u, s2u, lo, hi)
            if self.learn_items:
                mu, w = lik()
                hJ = np.bincount(J, w, nJ) + 1 / s2d
                pm = cb + lam_d * yc
                d += (-np.bincount(J, y - mu, nJ) - (d - pm) / s2d) / hJ
                vd = 1 / hJ
                if learn_d and it >= self.em_start:
                    lam_d, s2d = self._hyper(d - cb, vd, yc, var_d, self.nu_d, self.tau_d, s2d, lo, hi)
        _, w = lik()
        vb = 1 / (w.sum() + 1 / var_b) if self.learn_bench else var_b
        vdel = 1 / (np.bincount(P, w, nP) + 1 / s2u)
        vd = 1 / (np.bincount(J, w, nJ) + 1 / s2d) if self.learn_items else np.full(nJ, s2d)
        return {"beta": (float(beta), float(vb)),
                "delta": {sk: (float(delta[k]), float(vdel[k])) for sk, k in pidx.items()},
                "d": {ik: (float(d[k]), float(vd[k])) for ik, k in iidx.items()},
                "vb": float(vb),
                "hyp": {"lam_u": lam_u, "s2u": s2u, "lam_d": lam_d, "s2d": s2d, "cb": cb}}

    def score_batch(self, state, inputs):
        if self.subject_sharing:
            return super().score_batch(state, inputs)
        var_b, var_u, var_d = self._prior_vars()
        yh = self.item_priors([i for _, i in inputs])
        m, v = np.empty(len(inputs)), np.empty(len(inputs))
        fits = state["fits"]
        for k, (s, i) in enumerate(inputs):
            st = fits.get(i.get("benchmark_id", ""))
            if st:
                h = st["hyp"]
                bm, bv = st["beta"]
                dm, dv = st["delta"].get(subject_key(s), (None, None))
                if dm is None:
                    dm, dv = h["lam_u"] * self.subject_prior(s), h["s2u"]
                im, iv = st["d"].get(item_key(i), (None, None))
                if im is None:
                    im, iv = h["cb"] + h["lam_d"] * (yh[k] - h["cb"]), h["s2d"]
            else:
                bm, bv = self.mu_beta, var_b
                dm, dv = self.subject_prior(s), var_u
                im, iv = yh[k], var_d
            m[k], v[k] = bm + dm - im, bv + dv + iv
        v = v * self.score_var_mult
        return sigmoid(m / np.sqrt(1 + np.pi * v / 8)) if self.marginalize else sigmoid(m)
