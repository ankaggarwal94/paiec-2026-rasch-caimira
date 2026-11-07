"""Reference models from the collaborator's repo (cs321m-predictive-eval): CaimiraLite, retrained on our folds.

Static (they ignore `labeled`): logit from subject identity, benchmark, condition and a frozen mpnet item embedding
(`features/mpnet.py`, precomputed by scripts/embed_items.py). Two architectures and their logit ensemble:
  lite  (their m5 'modal CaimiraLite')   logit = theta_s . (W e) + b_s + b_bench + b_cond
  irtm  (their m3 'CaimiraLiteModel')    logit = sum_k (theta_sk - diff_k(e)) * softmax(rel(e))_k + c + biases
  ens   (their ensemble_c02)             mean of the two logits / temperature (1.10)
Training: BCE, Adam, early stopping on a held-out set of train items; torch is fit-time only, the fitted object holds
numpy weights. Unseen subjects / benchmarks / conditions at test time use the mean of the learned rows (their UNK slot
is an untrained random row; this is deliberate and noted in docs/research/ref.md).
"""
import numpy as np

from ..features.mpnet import EmbeddingTable
from ..features.subject import subject_identity
from .base import Predictor
from .simple import sigmoid


def _pick_device(torch):
    """cuda only if it can actually run a kernel: is_available() is True on GPUs the installed torch build has no
    kernels for (e.g. a Pascal card with an sm_70+ wheel), which then fails on the first op."""
    if torch.cuda.is_available():
        try:
            (torch.ones(2, device="cuda") * 2).sum().item()
            return "cuda"
        except Exception:  # noqa: BLE001 - any CUDA failure means fall back to cpu
            return "cpu"
    return "cpu"


def _softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


class RefCaimira(Predictor):
    name = "ref_caimira"
    RUNTIME_PARAMS = ("temperature", "unk")
    kind = "lite"           # lite | irtm | ens
    latent_dim = 5          # lite; irtm uses irtm_latent
    irtm_latent = 3
    epochs = 12
    lr = 1e-3
    batch = 8192
    val_frac = 0.05
    patience = 2
    seed = 0
    max_rows = 0            # >0: subsample training responses (smoke tests)
    temperature = 1.0       # ens: 1.10 in their champion
    unk = "mean"

    # ---------------- fit (torch) ----------------
    def fit(self, train):
        if self.kind == "ens":
            self.members = []
            for kind, ld in (("lite", self.latent_dim), ("irtm", self.irtm_latent)):
                m = RefCaimira(kind=kind, latent_dim=ld, irtm_latent=ld, epochs=self.epochs, lr=self.lr,
                               batch=self.batch, seed=self.seed, max_rows=self.max_rows, val_frac=self.val_frac,
                               patience=self.patience, unk=self.unk).fit(train)
                self.members.append(m)
            return self
        import torch
        rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        dev = _pick_device(torch)
        ident = [subject_identity(s) for s in train.subjects]
        s_names = sorted({ident[s] for s in np.unique(train.resp_s)})
        self.s_map = {n: k for k, n in enumerate(s_names)}
        b_names = sorted(set(train.resp_b.tolist()))
        self.b_map = {train.bench_names[b]: k for k, b in enumerate(b_names)}
        conds = sorted({train.items[i].get("interactors", "") or "none" for i in np.unique(train.resp_i)})
        self.c_map = {c: k for k, c in enumerate(conds)}
        ri, rs, ry = train.resp_i, train.resp_s, train.resp_y
        if self.max_rows and len(ry) > self.max_rows:
            keep = rng.choice(len(ry), self.max_rows, replace=False)
            ri, rs, ry = ri[keep], rs[keep], ry[keep]
        uitems = np.unique(ri)
        table = EmbeddingTable()
        E = table.lookup([train.items[i] for i in uitems], dtype=np.float16)
        pos = np.searchsorted(uitems, ri)
        S = np.array([self.s_map[ident[s]] for s in rs])
        B = np.array([self.b_map[train.bench_names[train.item_bench[i]]] for i in ri])
        C = np.array([self.c_map[train.items[i].get("interactors", "") or "none"] for i in ri])
        hold = rng.random(len(uitems)) < self.val_frac     # held-out items (item-heldout early stopping)
        is_val = hold[pos]
        Et = torch.tensor(E, dtype=torch.float16, device=dev)
        t = lambda a, dt=torch.long: torch.tensor(a, dtype=dt, device=dev)  # noqa: E731
        Ts, Tb, Tc, Tp, Ty = t(S), t(B), t(C), t(pos), t(ry, torch.float32)
        tr_idx, va_idx = t(np.nonzero(~is_val)[0]), t(np.nonzero(is_val)[0])
        ns, nb, nc = len(s_names), len(b_names), len(conds)
        D = E.shape[1]
        L = self.latent_dim if self.kind == "lite" else self.irtm_latent
        mod = _build(self.kind, ns, nb, nc, D, L).to(dev)
        opt = torch.optim.Adam(mod.parameters(), lr=self.lr)
        best, best_state, bad = 1e9, None, 0
        for ep in range(self.epochs):
            mod.train()
            perm = tr_idx[torch.randperm(len(tr_idx), device=dev)]
            for k in range(0, len(perm), self.batch):
                j = perm[k:k + self.batch]
                logit = mod(Ts[j], Tb[j], Tc[j], Et[Tp[j]].float())
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, Ty[j])
                opt.zero_grad()
                loss.backward()
                opt.step()
            mod.eval()
            with torch.no_grad():
                vl = 0.0
                for k in range(0, len(va_idx), 65536):
                    j = va_idx[k:k + 65536]
                    vl += torch.nn.functional.binary_cross_entropy_with_logits(
                        mod(Ts[j], Tb[j], Tc[j], Et[Tp[j]].float()), Ty[j], reduction="sum").item()
                vl /= max(len(va_idx), 1)
            print(f"    ref_caimira[{self.kind}] epoch {ep + 1} val_nll {vl:.4f}", flush=True)
            if len(va_idx) == 0 or vl < best - 1e-4:   # no validation rows: no early stopping, keep the last epoch
                best, bad = vl, 0
                best_state = {k: v.detach().cpu().clone() for k, v in mod.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        mod.load_state_dict(best_state)
        self.W = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in mod.state_dict().items()}
        self.val_nll = best
        return self

    # ---------------- runtime (numpy) ----------------
    def adapt(self, labeled):
        return None

    def extend(self, state, new):
        return None

    def _table(self):
        if not hasattr(self, "_cache_tab"):
            self._cache_tab = EmbeddingTable()
        return self._cache_tab

    def _row(self, W, key, idx, n):
        """Embedding-row lookup with UNK -> mean row."""
        mean = W[key].mean(0) if self.unk == "mean" else np.zeros(W[key].shape[1], np.float32)
        tab = np.vstack([W[key], mean[None]])
        return tab[np.where(idx < 0, n, idx)]

    def _logits(self, inputs):
        if self.kind == "ens":
            return np.mean([m._logits(inputs) for m in self.members], 0)
        W = self.W
        subj = [s for s, _ in inputs]
        items = [i for _, i in inputs]
        si = np.array([self.s_map.get(subject_identity(s), -1) for s in subj])
        bi = np.array([self.b_map.get(i.get("benchmark_id", ""), -1) for i in items])
        ci = np.array([self.c_map.get(i.get("interactors", "") or "none", -1) for i in items])
        e = self._table().lookup(items)
        ns, nb, nc = len(self.s_map), len(self.b_map), len(self.c_map)
        bias = (self._row(W, "subject_bias.weight", si, ns)[:, 0] + self._row(W, "benchmark_bias.weight", bi, nb)[:, 0]
                + self._row(W, "condition_bias.weight", ci, nc)[:, 0])
        if self.kind == "lite":
            theta = self._row(W, "subject_latent.weight", si, ns)
            return (theta * (e @ W["item_proj.weight"].T)).sum(-1) + bias
        theta = self._row(W, "subject_skill.weight", si, ns)
        rel = _softmax(e @ W["relevance.weight"].T + W["relevance.bias"])
        dr = e @ W["difficulty.weight"].T + W["difficulty.bias"]
        diff = dr - dr.mean(-1, keepdims=True)
        return ((theta - diff) * rel).sum(-1) + float(W["global_intercept"]) + bias

    def score_batch(self, state, inputs):
        return sigmoid(self._logits(inputs) / (self.temperature if self.kind == "ens" else 1.0))

    def score(self, state, subject, item):
        return float(self.score_batch(state, [(subject, item)])[0])


def _build(kind, ns, nb, nc, D, L):
    import torch
    from torch import nn

    class Lite(nn.Module):
        def __init__(self):
            super().__init__()
            self.subject_latent, self.subject_bias = nn.Embedding(ns, L), nn.Embedding(ns, 1)
            self.benchmark_bias, self.condition_bias = nn.Embedding(nb, 1), nn.Embedding(nc, 1)
            self.item_proj = nn.Linear(D, L, bias=False)
            for e in (self.subject_bias, self.benchmark_bias, self.condition_bias):
                nn.init.zeros_(e.weight)
            nn.init.normal_(self.subject_latent.weight, std=0.1)

        def forward(self, s, b, c, e):
            return ((self.subject_latent(s) * self.item_proj(e)).sum(-1) + self.subject_bias(s)[:, 0]
                    + self.benchmark_bias(b)[:, 0] + self.condition_bias(c)[:, 0])

    class IRTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.subject_skill = nn.Embedding(ns, L)
            self.relevance, self.difficulty = nn.Linear(D, L), nn.Linear(D, L)
            self.global_intercept = nn.Parameter(torch.zeros(()))
            self.subject_bias, self.benchmark_bias = nn.Embedding(ns, 1), nn.Embedding(nb, 1)
            self.condition_bias = nn.Embedding(nc, 1)
            for e in (self.subject_bias, self.benchmark_bias, self.condition_bias):
                nn.init.zeros_(e.weight)
            nn.init.normal_(self.subject_skill.weight, std=0.1)

        def forward(self, s, b, c, e):
            rel = torch.softmax(self.relevance(e), -1)
            dr = self.difficulty(e)
            diff = dr - dr.mean(-1, keepdim=True)
            return (((self.subject_skill(s) - diff) * rel).sum(-1) + self.global_intercept + self.subject_bias(s)[:, 0]
                    + self.benchmark_bias(b)[:, 0] + self.condition_bias(c)[:, 0])

    return Lite() if kind == "lite" else IRTM()
