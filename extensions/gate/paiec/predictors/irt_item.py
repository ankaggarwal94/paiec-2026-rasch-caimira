"""IRT with online, within-benchmark item-difficulty amortisation (test time only).

    logit p(s, i) = beta_b + delta_{s,b} - d_i
    d_i = yhat_i + z_i . w_b + sum_k a_ik g_{b,k} + eps_i

- yhat_i: the cross-benchmark content prior of IRT (unchanged fit).
- z_i . w_b: a per-benchmark linear map on content features, learned from the benchmark's revealed labels.
  z_i is a seeded random projection (rp_dim) of the IRT item features, centred on the benchmark's labeled
  items and normalised to unit length (a cosine kernel). w_b ~ N(0, rp_frac * var_d I).
- g_{b,k}: group effects for each item_features key=value token (kv_frac) and for identical item_content
  (dup_frac), g ~ N(0, 1) with weights a_ik = sqrt(frac * var_d / n_keys(i)).
- eps_i ~ N(0, (1 - rp_frac - kv_frac - dup_frac) * var_d): the item's own effect, as in IRT.
The prior variance of every unlabeled item therefore stays var_d (budget 0 is unchanged); labels on other
items of the same benchmark now move unlabeled items' difficulties through shared content and groups.

All new parameters are test-time only (RUNTIME_PARAMS), and the fit is IRT's, so the IRT fit cache is
reused (FIT_AS). Runtime: numpy + stdlib.
"""
import zlib

import numpy as np

from ..features.hashing import parse_kv
from ..features.subject import subject_key
from .base import Evidence, item_key
from .irt import IRT
from .irt_lr import IRTLR
from .simple import sigmoid



def _solve(A, b):
    """np.linalg.solve with a least-squares fallback for singular systems (degenerate benchmarks: few items, collinear features)."""
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]


def _inv(A):
    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A)


class IRTItem(IRT):
    name = "irt_item"
    FIT_AS = IRT
    RUNTIME_PARAMS = IRT.RUNTIME_PARAMS + ("rp_frac", "rp_dim", "rp_seed", "kv_frac", "dup_frac", "item_iters",
                                           "kv_numeric", "eb_nu", "eb_eps_nu", "rp2_frac", "rp2_dim", "prior_slope_var", "vi", "vi_frac", "lr_on",
                                           "eb_u_nu", "vi_parts", "eb_bound")
    rp_frac = 0.0
    # online-EM scales (eb_nu) stay within [1/eb_bound, eb_bound] x their configured value. Without a bound, a small
    # near-separable benchmark can run away (bigger weights -> bigger scale -> weaker prior) until H is singular.
    eb_bound = 100.0
    rp_dim = 64
    rp_seed = 0
    # degree-2 block: a tensor sketch of (z_i . z_j)^2 on the centred, normalised map features (a sharper,
    # more local kernel that emphasises near-duplicates); its own variance fraction. Requires rp_frac > 0.
    rp2_frac = 0.0
    # per-benchmark slope on the cross-benchmark prior: d += c_b * (yhat_i - mean yhat), c_b ~ N(0, this).
    # Lets a benchmark learn to trust (c > 0), ignore (c = -1) or flip its content prior. Requires rp_frac > 0.
    prior_slope_var = 0.0
    rp2_dim = 64
    kv_frac = 0.0
    dup_frac = 0.0
    kv_numeric = False      # also group on numeric kv values
    item_iters = 0          # >0 overrides adapt_iters for this model
    # online empirical Bayes: per-benchmark scales of the kv-key and content-map variances, learned by EM with
    # a pseudo-count eb_nu (groups / map dims) toward the configured fraction. 0 = fixed fractions.
    eb_nu = 0.0
    eb_eps_nu = 0.0         # same for the item-specific variance (pseudo-count in items); 0 = fixed
    # mean-field Gaussian VI instead of the joint MAP (as in irt_h): Newton steps use the expected likelihood
    # sigmoid(eta / kappa), kappa = sqrt(1 + pi * vi_frac * v / 8), v = summed current posterior variances.
    vi = False
    vi_frac = 1.0
    lr_on = False           # subject-descriptor regression (IRTItemLR): delta_s ~ N(x_s . w_b, var_r)
    # online EB for the per-benchmark pair-ability variance (as irt_h nu_u): EM with pseudo-count eb_u_nu (pairs)
    # toward the trained var_u (or var_r with lr_on), clipped to [0.05, 20] x that value. 0 = fixed.
    eb_u_nu = 0.0
    vi_parts = "bud"        # which posterior variances enter the VI expectation: b(eta offset), u(pair), d(item)

    # ---------------- item info (cached, pure) ----------------
    def _sig(self):
        """Parameters the cached item info depends on. Mix members may override them, so every cache is keyed by it."""
        return (self.rp_frac > 0, int(self.rp_dim), int(self.rp_seed), bool(self.kv_numeric),
                bool(getattr(self, "emb_online", False)))

    def _poly(self, Z):
        key = (Z.shape[-1], int(self.rp2_dim), int(self.rp_seed))
        P = getattr(self, "_cache_P", None)
        if P is None or P[0] != key:
            rng = np.random.default_rng(int(self.rp_seed) + 1)
            P = self._cache_P = (key, rng.standard_normal((Z.shape[-1], int(self.rp2_dim))),
                                 rng.standard_normal((Z.shape[-1], int(self.rp2_dim))))
        return (Z @ P[1]) * (Z @ P[2]) / np.sqrt(self.rp2_dim)

    def _dense(self, zc):
        return np.concatenate([zc, self._poly(zc)], -1) if self.rp2_frac > 0 else zc

    def online_features(self, items, X):
        """Feature vectors for the online content map (one row per item). Default: the IRT prior's own
        features X (ItemFeaturizer). Override or swap the featurizer here to use another representation."""
        return X

    def _proj(self, n_in):
        key = (n_in, int(self.rp_dim), int(self.rp_seed))
        R = getattr(self, "_cache_R", None)
        if R is None or R[0] != key:
            rng = np.random.default_rng(int(self.rp_seed))
            R = self._cache_R = (key, rng.standard_normal((n_in, int(self.rp_dim))) / np.sqrt(self.rp_dim))
        return R[1]

    def _kv_tokens(self, item):
        out = []
        for k, v in parse_kv(item.get("item_features", "")).items():
            if not v:
                continue
            if not self.kv_numeric:
                try:
                    float(v)
                    continue
                except ValueError:
                    pass
            out.append("kv:" + k + "=" + v)
        return tuple(out)

    def _info(self, items):
        """Per item: (prior mean, projection z, kv tokens, content key). Cached by item key."""
        if not hasattr(self, "_cache_info"):
            self._cache_info = {}
        ci = self._cache_info.setdefault(self._sig(), {})
        keys = [item_key(i) for i in items]
        todo = {}
        for k, i in zip(keys, items):
            if k not in ci and k not in todo:
                todo[k] = i
        if todo:
            X = self.ifeat.transform(list(todo.values()))
            yh = X @ self.w_i + self.b_i
            if self.rp_frac > 0:
                F = self.online_features(list(todo.values()), X)
                Z = F.astype(np.float64) @ self._proj(F.shape[1])
            else:
                Z = np.zeros((len(X), 0))
            for n, (k, i) in enumerate(todo.items()):
                ck = "ct:%08x" % zlib.crc32(i.get("item_content", "").encode())
                ci[k] = (float(yh[n]), Z[n], self._kv_tokens(i), ck)
        return [ci[k] for k in keys]

    def item_priors(self, items) -> np.ndarray:
        return np.array([x[0] for x in self._info(items)], float)

    def _groups(self, info):
        """[(group token, weight)] for one item."""
        vd = self.var_d * self.var_scale
        out = []
        if self.kv_frac > 0 and info[2]:
            a = np.sqrt(self.kv_frac * vd / len(info[2]))
            out += [(t, a) for t in info[2]]
        if self.dup_frac > 0:
            out.append((info[3], np.sqrt(self.dup_frac * vd)))
        return out

    def _var_eps(self):
        return max(1.0 - self.rp_frac - self.rp2_frac - self.kv_frac - self.dup_frac, 0.02) * self.var_d * self.var_scale

    # ---------------- adapt ----------------
    def extend(self, state, new):
        assert not self.subject_sharing, "irt_item supports per-benchmark adaptation only"
        touched = state["ev"].add(new)
        for b in touched:
            state["fits"][b] = self._safe_adapt_bench(state["ev"].by_bench[b], state["fits"].get(b))
        return state

    def _safe_adapt_bench(self, obs, warm=None):
        """One benchmark's fit, or None (score falls back to the priors) if it fails or is not finite. Evidence is
        global, so an exception here would otherwise lose every benchmark's predictions at this checkpoint."""
        try:
            out = self._adapt_bench(obs, warm)
        except (np.linalg.LinAlgError, FloatingPointError, ValueError, ZeroDivisionError):
            return None
        vals = [*out["beta"], *(x for dv in out["delta"].values() for x in dv), *(x for dv in out["d"].values() for x in dv)]
        if out.get("w") is not None:
            vals += [float(np.abs(out["w"]).max(initial=0.0))]
        return out if np.all(np.isfinite(vals)) else None

    def adapt(self, labeled):
        return self.extend({"ev": Evidence(), "fits": {}, "joint": None}, labeled)

    def _adapt_bench(self, obs, warm=None):
        vs = self.var_scale
        var_b, var_u = self.var_beta * vs, self.var_u * vs
        var_e = self._var_eps()
        tau2 = self.rp_frac * self.var_d * vs
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
        lr = self.lr_on
        if lr:
            X = np.array([self.lr_x(s) for s in subj])
            var_u = self.lr_var_r * self.lr_rvar_mult * vs
            Si = _inv(self.lr_S * self.lr_wvar_mult * vs)
            lw = self.lr_wbar.copy()
            u = X @ lw
        else:
            u = np.array([self.subject_prior(s) for s in subj])
        info = self._info(items)
        yh = np.array([x[0] for x in info])
        nP, nJ = len(u), len(yh)

        # shared content map
        use_rp = tau2 > 0
        if use_rp:
            Z = np.stack([x[1] for x in info])
            zbar = Z.mean(0)
            Zc = Z - zbar
            Zc /= np.maximum(np.linalg.norm(Zc, axis=1, keepdims=True), 1e-9)
            k1 = Zc.shape[1]
            Zc = self._dense(Zc)
            k2 = Zc.shape[1]
            ybar = float(yh.mean())
            if self.prior_slope_var > 0:
                Zc = np.concatenate([Zc, (yh - ybar)[:, None]], 1)
            k = Zc.shape[1]
            slices = [(slice(0, k1), tau2)]
            if k2 > k1:
                slices.append((slice(k1, k2), self.rp2_frac * self.var_d * vs))
            if k > k2:
                slices.append((slice(k2, k), float(self.prior_slope_var)))
            w = np.zeros(k)
        # groups: one block per kv key (disjoint values within a key) + one content block
        blocks = {}
        gidx = {}
        bkeys = []
        for j, x in enumerate(info):
            for t, a in self._groups(x):
                bk = t.split("=", 1)[0] if t.startswith("kv:") else "ct"
                gi = gidx.setdefault(t, len(gidx))
                if bk not in blocks:
                    blocks[bk] = ([], [], [])
                    bkeys.append(bk)
                blocks[bk][0].append(j)
                blocks[bk][1].append(gi)
                blocks[bk][2].append(a)
        nG = len(gidx)
        g = np.zeros(nG)
        blocks = [(np.array(jj), np.array(gg), np.array(aa)) for jj, gg, aa in (blocks[b] for b in bkeys)]
        s2 = np.ones(len(blocks))
        bsel = []
        for jj, gg, aa in blocks:
            sel = np.zeros(nG, bool)
            sel[gg] = True
            bsel.append(sel)
        t2 = [tv for _, tv in slices] if use_rp else []
        # item-level map from obs to the items' group contributions: shared(j) = sum a * g
        beta, delta, eps = self.mu_beta, u.copy(), np.zeros(nJ)
        Apost = Si if lr else None
        if warm:
            beta = warm["beta"][0]
            if lr:
                lw = warm["lr_w"].copy()
                u = X @ lw
            for sk, kk in pidx.items():
                if sk in warm["delta"]:
                    delta[kk] = warm["delta"][sk][0]
            we = warm.get("eps", {})
            for ik, kk in iidx.items():
                if ik in we:
                    eps[kk] = we[ik]
            if use_rp and warm.get("w") is not None and len(warm["w"]) == k:
                w = warm["w"].copy()
            if use_rp and warm.get("t2") and len(warm["t2"]) == len(t2):
                t2 = list(warm["t2"])
            ws2 = warm.get("gs2", {})
            for bi, bk in enumerate(bkeys):
                s2[bi] = ws2.get(bk, 1.0)
            wg = warm.get("g", {})
            for t, gi in gidx.items():
                if t in wg:
                    g[gi] = wg[t][0]

        def shared():
            s = np.zeros(nJ)
            if use_rp:
                s += Zc @ w
            for jj, gg, aa in blocks:
                s += np.bincount(jj, aa * g[gg], nJ)
            return s

        sh = shared()
        vd_tot = self.var_d * vs
        vb_, vdel_, vd_ = var_b, np.full(nP, var_u), np.full(nJ, vd_tot)

        def lik():
            eta = beta + delta[P] - d[J]
            if not self.vi:
                mu = sigmoid(eta)
                return mu, mu * (1 - mu)
            vp = self.vi_parts
            v = ((vb_ if self.learn_bench and "b" in vp else 0.0) + (vdel_[P] if "u" in vp else 0.0)
                 + (vd_[J] if "d" in vp else 0.0))
            kap = np.sqrt(1 + np.pi * self.vi_frac * v / 8)
            mu = sigmoid(eta / kap)
            return mu, mu * (1 - mu) / kap

        nu_u, var_u0 = float(self.eb_u_nu), var_u
        if warm and nu_u > 0 and warm.get("var_u"):
            var_u = warm["var_u"]
        iters = int(self.item_iters) or self.adapt_iters
        nu = float(self.eb_nu)
        nu_e, var_e0 = float(self.eb_eps_nu), var_e
        if warm and nu_e > 0 and warm.get("var_e"):
            var_e = warm["var_e"]
        for _ in range(iters):
            d = yh + sh + eps
            if self.learn_bench:
                mu, wt = lik()
                hb = wt.sum() + 1 / var_b
                beta += ((y - mu).sum() - (beta - self.mu_beta) / var_b) / hb
                vb_ = 1 / hb
            mu, wt = lik()
            hP = np.bincount(P, wt, nP) + 1 / var_u
            delta += (np.bincount(P, y - mu, nP) - (delta - u) / var_u) / hP
            vdel_ = 1 / hP
            if nu_u > 0:
                var_u = float(np.clip((((delta - u) ** 2 + vdel_).sum() + nu_u * var_u0) / (nP + nu_u),
                                      0.05 * var_u0, 20 * var_u0))
            if lr and self.lr_adapt_w:
                hq = np.bincount(P, wt, nP) + 1e-9   # likelihood curvature (hP carries the pre-update var_u)
                zs = delta + np.bincount(P, y - mu, nP) / hq
                wq = 1.0 / (var_u + 1.0 / hq)
                Apost = X.T @ (X * wq[:, None]) + Si
                lw = _solve(Apost, X.T @ (wq * zs) + Si @ self.lr_wbar)
                u = X @ lw
            if self.learn_items:
                mu, wt = lik()
                h = np.bincount(J, wt, nJ)
                vd_ = 1 / (h + 1 / vd_tot)
                eps += (-np.bincount(J, y - mu, nJ) - eps / var_e) / (h + 1 / var_e)
                if nu_e > 0:
                    var_e = (float((eps ** 2 + 1 / (h + 1 / var_e)).sum()) + nu_e * var_e0) / (nJ + nu_e)
                d = yh + sh + eps
                if use_rp:
                    mu, wt = lik()
                    r, h = np.bincount(J, y - mu, nJ), np.bincount(J, wt, nJ)
                    pv = np.concatenate([np.full(sl.stop - sl.start, tv) for (sl, _), tv in zip(slices, t2)])
                    H = (Zc * h[:, None]).T @ Zc + np.diag(1 / pv)
                    w += _solve(H, -Zc.T @ r - w / pv)
                    if nu > 0:
                        he = h / (1 + h * var_e)
                        Sm = _inv((Zc * he[:, None]).T @ Zc + np.diag(1 / pv))
                        t2 = [float(np.clip((w[sl] @ w[sl] + np.trace(Sm[sl, sl]) + nu * t0) / (sl.stop - sl.start + nu),
                                            t0 / self.eb_bound, t0 * self.eb_bound))
                              for sl, t0 in slices]
                    sh = shared()
                    d = yh + sh + eps
                for bi, (jj, gg, aa) in enumerate(blocks):
                    mu, wt = lik()
                    r, h = np.bincount(J, y - mu, nJ), np.bincount(J, wt, nJ)
                    sel = bsel[bi]
                    step = (np.bincount(gg, -aa * r[jj], nG) - g / s2[bi]) / (np.bincount(gg, aa * aa * h[jj], nG) + 1 / s2[bi])
                    g[sel] += step[sel]
                    if nu > 0:
                        he = h / (1 + h * var_e)
                        vgb = 1 / (np.bincount(gg, aa * aa * he[jj], nG) + 1 / s2[bi])
                        s2[bi] = min(max((float((g[sel] ** 2 + vgb[sel]).sum()) + nu) / (sel.sum() + nu),
                                         1 / self.eb_bound), self.eb_bound)
                    sh = shared()
                    d = yh + sh + eps
        d = yh + sh + eps
        _, wt = lik()
        h = np.bincount(J, wt, nJ)
        vb = 1 / (wt.sum() + 1 / var_b) if self.learn_bench else var_b
        vdel = 1 / (np.bincount(P, wt, nP) + 1 / var_u)
        if lr:
            hq = np.bincount(P, wt, nP)
            xs = np.einsum("ij,jk,ik->i", X, _inv(Apost), X)
            vdel = 1 / (hq + 1 / var_u) + ((1 / var_u) / (hq + 1 / var_u)) ** 2 * xs
        # group posterior variances (diagonal), shared-part variance per labeled item
        vg = np.ones(nG)
        shv = np.zeros(nJ)
        heff = h / (1 + h * var_e)
        for bi, (jj, gg, aa) in enumerate(blocks):
            vg_b = 1 / (np.bincount(gg, aa * aa * heff[jj], nG) + 1 / s2[bi])
            vg[bsel[bi]] = vg_b[bsel[bi]]
        for jj, gg, aa in blocks:
            shv += np.bincount(jj, aa * aa * vg[gg], nJ)
        out = {"beta": (float(beta), float(vb)),
               "delta": {sk: (float(delta[kk]), float(vdel[kk])) for sk, kk in pidx.items()},
               "eps": {ik: float(eps[kk]) for ik, kk in iidx.items()},
               "g": {t: (float(g[gi]), float(vg[gi])) for t, gi in gidx.items()},
               **({"lr_w": lw, "lr_S": _inv(Apost)} if lr else {}),
               "gs2": {bk: float(s2[bi]) for bi, bk in enumerate(bkeys)}, "var_e": float(var_e),
               **({"var_u": float(var_u)} if nu_u > 0 else {})}
        if use_rp:
            pv = np.concatenate([np.full(sl.stop - sl.start, tv) for (sl, _), tv in zip(slices, t2)])
            Sw = _inv((Zc * heff[:, None]).T @ Zc + np.diag(1 / pv))
            shv += np.einsum("ij,jk,ik->i", Zc, Sw, Zc)
            out.update(w=w, Sw=Sw, zbar=zbar, ybar=ybar, t2=[float(t) for t in t2])
        else:
            out.update(w=None)
        vd = 1 / (h + 1 / (var_e + shv)) if self.learn_items else var_e + shv
        out["d"] = {ik: (float(d[kk]), float(vd[kk])) for ik, kk in iidx.items()}
        return out

    # ---------------- score ----------------
    def _item_post(self, st, inf):
        """(mean, var) of an unlabeled item's difficulty given a benchmark state (or None)."""
        m, v = inf[0], (st["var_e"] if st else self._var_eps())
        for t, a in self._groups(inf):
            prior = (0.0, st["gs2"].get(t.split("=", 1)[0] if t.startswith("kv:") else "ct", 1.0)) if st else (0.0, 1.0)
            gm, gv = st["g"].get(t, prior) if st else prior
            m += a * gm
            v += a * a * gv
        if self.rp_frac > 0:
            if st and st.get("w") is not None:
                zc = inf[1] - st["zbar"]
                zc = self._dense(zc / max(np.linalg.norm(zc), 1e-9))
                if self.prior_slope_var > 0:
                    zc = np.append(zc, inf[0] - st["ybar"])
                m += float(zc @ st["w"])
                v += float(zc @ st["Sw"] @ zc)
            else:
                v += (self.rp_frac + self.rp2_frac) * self.var_d * self.var_scale
        return m, v

    def score_batch(self, state, inputs):
        vs = self.var_scale
        infos = self._info([i for _, i in inputs])
        m, v = np.empty(len(inputs)), np.empty(len(inputs))
        fits = state["fits"]
        for k, (s, i) in enumerate(inputs):
            st = fits.get(i.get("benchmark_id", ""))
            bm, bv = st["beta"] if st else (self.mu_beta, self.var_beta * vs)
            dm, dv = st["delta"].get(subject_key(s), (None, None)) if st else (None, None)
            if dm is None and self.lr_on:
                x = self.lr_x(s)
                var_r = (st or {}).get("var_u", self.lr_var_r * self.lr_rvar_mult * vs)
                w, S = (st["lr_w"], st["lr_S"]) if st else (self.lr_wbar, self.lr_S * self.lr_wvar_mult * vs)
                dm, dv = float(x @ w), var_r + float(x @ S @ x)
            elif dm is None:
                dm, dv = self.subject_prior(s), (st or {}).get("var_u", self.var_u * vs)
            im, iv = st["d"].get(item_key(i), (None, None)) if st else (None, None)
            if im is None:
                im, iv = self._item_post(st, infos[k])
            m[k], v[k] = bm + dm - im, bv + dv + iv
        return sigmoid(m / np.sqrt(1 + np.pi * v / 8)) if self.marginalize else sigmoid(m)


class IRTItemLR(IRTItem, IRTLR):
    """Online item amortisation + per-benchmark subject-descriptor regression (fit as IRTLR)."""
    name = "irt_item_lr"
    FIT_AS = None
    lr_on = True
    RUNTIME_PARAMS = IRTItem.RUNTIME_PARAMS + IRTLR.RUNTIME_PARAMS
