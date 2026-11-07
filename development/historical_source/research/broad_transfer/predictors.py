"""Cold-start predictors for the common broad transfer panel.

The only fitted data supplied to fit_predictor are the training indices and
their observed binary outcomes. FeatureBank applies a fixed, stateless hash
to each text independently and can also be constructed for new inputs.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import expit
from torch import nn
from torch.nn import functional as F

from research.proxy_gpu.models import _optimize, _parameters, _torch
from research.proxy_iteration.features import lexical_features
from research.proxy_structured import structured as irt
from submissions.ncf.ncf_head import NCFHead

METHODS = ("lexical", "rasch", "caimira", "ncf", "entitymarformer")
SCHEMA = "broad-transfer-predictor-v1"
WIDTH = 384
EPOCHS = 6
ENTITY_WIDTH = 32
BATCH = 1024


class FeatureBank:
    """Cache transform results per distinct string, retaining input alignment."""

    def __init__(self, rows):
        item_texts, subjects, pairs, groups = {}, {}, {}, {}
        indexes = np.empty((len(rows), 4), dtype=np.int64)
        for i, row in enumerate(rows):
            text, subject = row["text"], row["subject"]
            indexes[i] = (
                item_texts.setdefault(text, len(item_texts)),
                subjects.setdefault(subject, len(subjects)),
                pairs.setdefault((row["benchmark"], subject), len(pairs)),
                groups.setdefault((row["benchmark"], row["group"]), len(groups)),
            )
        self.item, self.subject, self.pair, self.group = indexes.T
        self.x = self._encode(item_texts)
        self.z = self._encode(subjects)

    @staticmethod
    def _encode(strings):
        keys = list(strings)
        matrix = np.empty((len(keys), WIDTH), dtype=np.float32)
        for start in range(0, len(keys), 256):
            matrix[start : start + 256] = lexical_features(keys[start : start + 256])
        return matrix


def _indices(indices, length):
    a = np.asarray(indices)
    if a.ndim != 1 or (a.size and (a.dtype.kind not in "iu" or a.min() < 0 or a.max() >= length)):
        raise ValueError("invalid feature row indices")
    return a.astype(np.int64, copy=False)


class Training:
    """Conditional Bernoulli sufficient statistics with equal total weight per pair."""

    def __init__(self, bank, indices, labels):
        ix = _indices(indices, len(bank.item))
        y = np.asarray(labels)
        if not len(ix) or y.shape != ix.shape or y.dtype.kind not in "iuf" or not np.isin(y, (0, 1)).all():
            raise ValueError("training requires aligned binary numeric labels")
        self.rows, inverse, self.count = np.unique(ix, return_inverse=True, return_counts=True)
        self.success = np.bincount(inverse, weights=y).astype(np.int64)
        self.rate = self.success / self.count
        pair, self.pair = np.unique(bank.pair[self.rows], return_inverse=True)
        self.trials = len(y)
        self.pairs = len(pair)
        counts = np.bincount(self.pair, weights=self.count)
        self.unit_weight = 1 / (self.pairs * counts[self.pair])
        self.weight = self.count * self.unit_weight


def _lexical(bank, train):
    texts, inv = np.unique(bank.item[train.rows], return_inverse=True)
    x = bank.x[texts].astype(np.float64)
    weight = np.bincount(inv, weights=train.weight)
    success = np.bincount(inv, weights=train.success * train.unit_weight)
    center = weight @ x
    x -= center
    ridge = 0.001

    def objective(beta):
        logits = x @ beta[1:] + beta[0]
        residual = weight * expit(logits) - success
        nll = weight @ np.logaddexp(0, logits) - success @ logits
        return nll + ridge / 2 * (beta @ beta), np.r_[residual.sum(), x.T @ residual] + ridge * beta

    result = minimize(
        objective,
        np.zeros(WIDTH + 1),
        method="L-BFGS-B",
        jac=True,
        options={"gtol": 1e-8, "ftol": 1e-12, "maxiter": 300},
    )
    if not result.success or not np.isfinite(result.fun):
        raise ArithmeticError(f"lexical fit unsuccessful: {result.message}")
    return {"center": center.tolist(), "beta": result.x.tolist()}, {
        "optimizer": "L-BFGS-B",
        "ridge": ridge,
        "iterations": result.nit,
        "objective": float(result.fun),
        "gradient_inf": float(np.abs(result.jac).max()),
    }


def _structured(method, bank, train, seed, device):
    """Use the project's native loss, gauges, optimizer, and export format.

    A repeated binary outcome contributes its multiplicity times its original
    per-trial weight. This avoids repeating the item/subject feature matrices.
    """
    library, dtype = _torch(device, "float32")
    item_keys, item = np.unique(bank.item[train.rows], return_inverse=True)
    subject_keys, subject = np.unique(bank.subject[train.rows], return_inverse=True)
    config = irt._config(method, 2, "metadata", WIDTH, WIDTH, "float32", 0.01, 0.25)
    rank = config["rank"]
    p = _parameters(library, irt._shapes(config, len(subject_keys)), seed, device, dtype)
    with torch.no_grad():
        p["intercept"].zero_()
        if method == "caimira":
            p["relevance_bias"].zero_()

    def tensor(value, kind=dtype):
        return torch.as_tensor(value, dtype=kind, device=device)

    fail = train.count - train.success
    rows = np.r_[np.flatnonzero(fail), np.flatnonzero(train.success)]
    labels = np.r_[np.zeros(np.count_nonzero(fail)), np.ones(np.count_nonzero(train.success))]
    weights = np.r_[(fail * train.unit_weight)[fail > 0], (train.success * train.unit_weight)[train.success > 0]]
    x, z = tensor(bank.x[item_keys]), tensor(bank.z[subject_keys])
    present = (z != 0).any(1)
    zcenter = z[present].mean(0) if present.any() else z.new_zeros(WIDTH)
    data = {
        "bank": x,
        "subject_bank": z,
        "subject_center": zcenter,
        "nuisance_basis": tensor(irt._helmert(len(subject_keys))),
        "item_index": tensor(item[rows], torch.int64),
        "subject_index": tensor(subject[rows], torch.int64),
        "labels": tensor(labels),
        "weights": tensor(weights),
    }
    diag = _optimize(library, p, lambda: irt.structured_objective(p, data, config), 300, 1200)
    with torch.no_grad():
        _, loadings, _, aux = irt._item_tensor(p, x, x, config)
        state = {
            "schema": irt.SCHEMA,
            "config": config,
            "parameters": {k: v.cpu().tolist() for k, v in p.items() if k != "nuisance"},
            "centers": {
                "difficulty": aux["difficulty_center"].cpu().tolist(),
                "feature": x.mean(0).cpu().tolist(),
                "subject": zcenter.cpu().tolist(),
                "contrast": aux["contrast_center"].cpu().tolist(),
                "discrimination_log": aux["discrimination_log_center"].cpu().tolist(),
            },
            "orientation": irt._orientation(p, loadings, config).tolist(),
        }
    return state, {
        "optimizer": "Adam",
        "rank": rank,
        "steps": 300,
        "convergence": diag["convergence"],
        **diag["history"][-1],
    }


class RelationalMarformer(nn.Module):
    """A compact binary Marformer adaptation with explicit subject and item entities.

    Each query attends to its subject, item and purchased response tokens.
    Each relation type has learned per-head key and value shifts. The masked
    binary query passes through two rounds of attention and a feedforward step.
    This is a new broad-panel variant of the relational architecture.
    """

    def __init__(self):
        super().__init__()
        width = ENTITY_WIDTH
        self.item = nn.Linear(WIDTH, width)
        self.subject = nn.Linear(WIDTH, width)
        self.observed = nn.Embedding(2, width)
        self.missing = nn.Parameter(torch.zeros(width))
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "q": nn.Linear(width, width),
                        "k": nn.Linear(width, width),
                        "v": nn.Linear(width, width),
                        "relation_k": nn.Embedding(3, width),
                        "relation_v": nn.Embedding(3, width),
                        "out": nn.Linear(width, width),
                        "attention_norm": nn.LayerNorm(width),
                        "ff": nn.Sequential(nn.Linear(width, 2 * width), nn.ReLU(), nn.Linear(2 * width, width)),
                        "ff_norm": nn.LayerNorm(width),
                    }
                )
                for _ in range(2)
            ]
        )
        self.out = nn.Linear(width, 1)

    def forward(self, item, subject, support, labels, observed_mask, pool):
        batch = len(item)
        context = torch.cat(
            (
                self.subject(subject)[:, None],
                self.item(item)[:, None],
                (self.item(support) + self.observed(labels))[pool],
            ),
            dim=1,
        )
        mask = torch.cat((torch.ones((batch, 2), dtype=torch.bool, device=item.device), observed_mask[pool]), dim=1)
        types = torch.arange(context.shape[1], device=item.device).clamp(max=2)
        query = self.missing.expand(batch, -1)
        heads, width = 4, ENTITY_WIDTH // 4
        for block in self.blocks:
            key = (block["k"](context) + block["relation_k"](types)).reshape(batch, -1, heads, width)
            value = (block["v"](context) + block["relation_v"](types)).reshape(batch, -1, heads, width)
            score = (block["q"](query).view(batch, 1, heads, width) * key).sum(-1) / width**0.5
            attention = torch.softmax(score.masked_fill(~mask[:, :, None], -torch.inf), dim=1)
            update = (attention[..., None] * value).sum(1).flatten(1)
            query = block["attention_norm"](query + block["out"](update))
            query = block["ff_norm"](query + block["ff"](query))
        return self.out(query).squeeze(-1)


def _network(method):
    return NCFHead(embed_dim=WIDTH, hidden=64) if method == "ncf" else RelationalMarformer()


def _training_histories(bank, train, texts, rng, budget):
    """Sample at most one observed trial per distinct text from other source groups."""
    n = len(train.rows)
    # Only a small set of candidate pools is needed: split a pair by source
    # group, and give its two sides disjoint purchase banks for this epoch.
    sides = np.zeros(n, dtype=np.int64)
    support = np.zeros((train.pairs * 2, budget), dtype=np.int64)
    labels = np.zeros_like(support)
    masks = np.zeros_like(support, dtype=bool)
    groups = bank.group[train.rows]
    if budget == 0:
        return sides, support, labels, masks
    order = np.argsort(train.pair, kind="stable")
    for pair, ix in enumerate(np.split(order, np.flatnonzero(np.diff(train.pair[order])) + 1)):
        group_ids = rng.permutation(np.unique(groups[ix]))
        left = np.isin(groups[ix], group_ids[::2])
        for side, select in enumerate((left, ~left)):
            pool = pair * 2 + side
            targets, donors = ix[select], rng.permutation(ix[~select])
            sides[targets] = pool
            if not len(targets) or not len(donors):
                continue
            # An exact text alias might have a different upstream group ID.
            donors = donors[~np.isin(texts[donors], np.unique(texts[targets]))]
            _, first = np.unique(texts[donors], return_index=True)
            chosen = donors[np.sort(first)[:budget]]
            k = len(chosen)
            support[pool, :k] = texts[chosen]
            labels[pool, :k] = rng.random(k) < train.rate[chosen]
            masks[pool, :k] = True
    return sides, support, labels, masks


def _train_network(method, bank, train, seed, device):
    _torch(device, "float32")
    torch.manual_seed(seed)
    model = _network(method).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01, foreach=False)
    rng = np.random.default_rng(seed)
    history = []
    rows = train.rows
    text_keys, texts = np.unique(bank.item[rows], return_inverse=True)
    subject_keys, subject_ids = np.unique(bank.subject[rows], return_inverse=True)
    item = torch.as_tensor(bank.x[text_keys], device=device)
    subject = torch.as_tensor(bank.z[subject_keys], device=device)
    xi, zi, target, weight = (
        torch.as_tensor(a, device=device)
        for a in (texts, subject_ids, train.rate.astype(np.float32), train.weight.astype(np.float32))
    )
    for epoch in range(EPOCHS):
        # Every trial contributes to the conditional binomial loss each epoch.
        budget = (0, 7, 1, 15, 3, 31)[epoch]
        if method == "entitymarformer":
            sides, support, labels, mask = _training_histories(bank, train, texts, rng, budget)
            support, labels, mask = (torch.as_tensor(a, device=device) for a in (support, labels, mask))
        shuffled = rng.permutation(len(rows))
        objective = torch.zeros((), device=device)
        for start in range(0, len(rows), BATCH):
            ix = shuffled[start : start + BATCH]
            index = torch.as_tensor(ix, device=device)
            xb, zb = item[xi[index]], subject[zi[index]]
            if method == "ncf":
                logits = model(torch.cat((zb, xb), dim=1))
            else:
                pools, inverse = np.unique(sides[ix], return_inverse=True)
                logits = model(
                    xb, zb, item[support[pools]], labels[pools], mask[pools], torch.as_tensor(inverse, device=device)
                )
            contribution = (
                F.binary_cross_entropy_with_logits(logits, target[index], reduction="none") * weight[index]
            ).sum()
            loss = contribution * (len(rows) / len(ix))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            objective += contribution.detach()
        history.append(float(objective.cpu()))
    if not np.isfinite(history).all():
        raise ArithmeticError("nonfinite neural training objective")
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}, {
        "optimizer": "AdamW",
        "epochs": EPOCHS,
        "batch": BATCH,
        "epoch_nll": history,
        "architecture": "NCFHead lexical hidden64"
        if method == "ncf"
        else "relational Marformer lexical width32 two blocks",
    }


class Predictor:
    def __init__(self, method, state, receipt, device):
        self.method, self.state, self.receipt, self.device = method, state, receipt, device
        self._bank = None
        if method in ("rasch", "caimira"):
            self.model = irt.ResidentStructured(state, device=device)
        elif method in ("ncf", "entitymarformer"):
            self.model = _network(method).to(device).eval()
            self.model.load_state_dict(state, strict=True)
        elif method != "lexical":
            raise ValueError("unknown predictor")

    def save(self, path):
        path = Path(path)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
            tmp = file.name
        try:
            torch.save({"schema": SCHEMA, "method": self.method, "state": self.state, "receipt": self.receipt}, tmp)
            os.replace(tmp, path)
        finally:
            Path(tmp).unlink(missing_ok=True)

    def predict(self, bank, indices):
        return self.predict_with_evidence(bank, indices, [], [])

    @torch.inference_mode()
    def prepare_prediction(self, bank, indices):
        """Cache frozen item heads and subject priors for supplied query attributes."""
        self._bank = bank
        self._texts, keys = self._lookup(bank.item[indices], len(bank.x))
        self._subjects, subjects = self._lookup(bank.subject[indices], len(bank.z))
        if self.method in ("rasch", "caimira"):
            params = [self.model.item_parameters(bank.x[keys[i : i + 4096]]) for i in range(0, len(keys), 4096)]
            self._offset = np.concatenate([row[0] for row in params])
            self._loading = np.concatenate([row[1] for row in params])
            self._prior = self.model.prior(bank.z[subjects])
        elif self.method == "entitymarformer":

            def projected(matrix, ids, layer):
                return torch.cat(
                    [
                        layer(torch.as_tensor(matrix[ids[i : i + 4096]], device=self.device))
                        for i in range(0, len(ids), 4096)
                    ]
                )

            self._item_encoding = projected(bank.x, keys, self.model.item)
            self._subject_encoding = projected(bank.z, subjects, self.model.subject)
            self._entity_context = [
                (self._key_value(b, self._subject_encoding, 0), self._key_value(b, self._item_encoding, 1))
                for b in self.model.blocks
            ]

    @staticmethod
    def _lookup(values, size):
        unique = np.unique(values)
        mapping = np.full(size, -1, dtype=np.int64)
        mapping[unique] = np.arange(len(unique))
        return mapping, unique

    @staticmethod
    def _key_value(block, context, relation):
        return tuple(
            (block[name](context) + block["relation_" + name].weight[relation]).reshape(-1, 4, ENTITY_WIDTH // 4)
            for name in ("k", "v")
        )

    def _entity(self, bank, query, support, labels):
        """Project shared purchased evidence once per layer, including both relation shifts."""
        support_ids = self._texts[bank.item[support]]
        support_encoding = self._item_encoding[support_ids] + self.model.observed(
            torch.as_tensor(labels, dtype=torch.int64, device=self.device)
        )
        support_values = [self._key_value(b, support_encoding, 2) for b in self.model.blocks]
        result = []
        for start in range(0, len(query), 4096):
            ix = query[start : start + 4096]
            subject_ids = torch.as_tensor(self._subjects[bank.subject[ix]], device=self.device)
            item_ids = torch.as_tensor(self._texts[bank.item[ix]], device=self.device)
            q = self.model.missing.expand(len(ix), -1)
            for block, ((sk, sv), (ik, iv)), (ek, ev) in zip(
                self.model.blocks, self._entity_context, support_values, strict=True
            ):
                h = block["q"](q).reshape(len(ix), 4, -1) / (ENTITY_WIDTH // 4) ** 0.5
                subject_score = (h * sk[subject_ids]).sum(-1, keepdim=True)
                item_score = (h * ik[item_ids]).sum(-1, keepdim=True)
                evidence_score = torch.einsum("bhd,shd->bhs", h, ek)
                w = torch.softmax(torch.cat((subject_score, item_score, evidence_score), dim=-1), dim=-1)
                observed = torch.einsum("bhs,shd->bhd", w[:, :, 2:], ev)
                v = (w[:, :, :1] * sv[subject_ids] + w[:, :, 1:2] * iv[item_ids] + observed).flatten(1)
                q = block["attention_norm"](q + block["out"](v))
                q = block["ff_norm"](q + block["ff"](q))
            result.append(self.model.out(q).flatten().sigmoid().cpu().numpy())
        return np.concatenate(result)

    @torch.inference_mode()
    def predict_with_evidence(self, bank, indices, support_indices, support_labels):
        query = _indices(indices, len(bank.item))
        support = _indices(support_indices, len(bank.item))
        labels = np.asarray(support_labels)
        if labels.shape != support.shape or not np.isin(labels, (0, 1)).all() or len(labels) > 31:
            raise ValueError("invalid purchased evidence")
        if not len(query):
            return np.empty(0)
        if len(labels) and len(np.unique(bank.pair[np.r_[query, support]])) != 1:
            raise ValueError("within-pair evidence must match the target pair")
        if self.method in ("rasch", "caimira", "entitymarformer"):
            rows = np.r_[query, support]
            if (
                self._bank is not bank
                or np.any(self._texts[bank.item[rows]] < 0)
                or np.any(self._subjects[bank.subject[rows]] < 0)
            ):
                self.prepare_prediction(bank, rows)
            if self.method == "entitymarformer":
                return self._check(self._entity(bank, query, support, labels))
            text = self._texts[bank.item[query]]
            ability = self._prior[self._subjects[bank.subject[query]]]
            if len(labels):
                bought = self._texts[bank.item[support]]
                ability = _map(
                    self._offset[bought].astype(np.float64),
                    self._loading[bought].astype(np.float64),
                    ability[0].astype(np.float64),
                    labels,
                )
            return self._check(expit(self._offset[text] + (self._loading[text] * ability).sum(-1)))
        out = []
        for start in range(0, len(query), BATCH):
            ix = query[start : start + BATCH]
            x, z = bank.x[bank.item[ix]], bank.z[bank.subject[ix]]
            if self.method == "lexical":
                beta = self.state["beta"]
                p = expit(beta[0] + (x - self.state["center"]) @ beta[1:])
            else:
                xb, zb = (torch.as_tensor(t, device=self.device) for t in (x, z))
                logits = self.model(torch.cat((zb, xb), dim=-1))
                p = logits.sigmoid().cpu().numpy()
            out.append(p)
        return self._check(np.concatenate(out))

    @staticmethod
    def _check(predictions):
        predictions = predictions.astype(np.float64)
        if not np.isfinite(predictions).all() or np.any((predictions < 0) | (predictions > 1)):
            raise ArithmeticError("invalid forecast")
        return predictions


def _map(offset, loading, mean, labels):
    """Solve the native Gaussian-prior logistic MAP in one or two dimensions."""

    def obj(a):
        eta = offset + loading @ a
        delta = a - mean
        return (np.logaddexp(0, eta) - labels * eta).sum() + 0.5 * (delta @ delta), loading.T @ (
            expit(eta) - labels
        ) + delta

    result = minimize(obj, mean, jac=True, method="L-BFGS-B", options={"gtol": 1e-8, "ftol": 1e-13})
    if not np.isfinite(result.fun) or (not result.success and np.abs(result.jac).max() > 1e-6):
        raise ArithmeticError("structured MAP did not converge")
    return result.x


def fit_predictor(method, bank, train_indices, labels, *, seed=17, device="cpu"):
    if method not in METHODS or type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("invalid predictor or seed")
    start = monotonic()
    train = Training(bank, train_indices, labels)
    if method == "lexical":
        state, diag = _lexical(bank, train)
    elif method in ("rasch", "caimira"):
        state, diag = _structured(method, bank, train, seed, device)
    else:
        state, diag = _train_network(method, bank, train, seed, device)
    receipt = {
        "method": method,
        "seed": seed,
        "device": "cpu" if method == "lexical" else device,
        "trials": train.trials,
        "input_rows": len(train.rows),
        "pairs": train.pairs,
        "fit_seconds": monotonic() - start,
        "diagnostics": diag,
    }
    return Predictor(method, state, receipt, device)


def load_predictor(path, *, device="cpu"):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data["schema"] != SCHEMA:
        raise ValueError("unsupported predictor checkpoint")
    return Predictor(data["method"], data["state"], data["receipt"], device)
