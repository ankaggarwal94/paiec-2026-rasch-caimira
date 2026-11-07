"""Inductive structured response models, with trial-preserving GPU training.

Indexed training abilities are nuisance variables only. Exported inference states
contain item heads, frozen training centers and semantic priors, never subject
IDs or ability lookup tables. Torch is imported lazily.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter

import numpy as np

from research.proxy_gpu.models import _optimize, _parameters, _torch

SCHEMA = "proxy-structured-inductive/v1"
KINDS = ("rasch", "twopl", "threepl_global", "threepl_item", "mirt", "caimira")
FLOOR_CAP = 0.25
FLOOR_SHRINKAGE = 0.1
ITEM_FLOOR_SHRINKAGE = 1.0
CAIMIRA_L1 = 1e-4


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _array(value, *, ndim, name):
    def boolean(part):
        if isinstance(part, (bool, np.bool_)):
            return True
        if isinstance(part, (list, tuple)):
            return any(boolean(v) for v in part)
        return isinstance(part, np.ndarray) and part.dtype.kind == "b"

    if boolean(value):
        raise ValueError(name + " cannot contain booleans")
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or raw.ndim != ndim or not np.isfinite(raw).all():
        raise ValueError(name + " must be a finite numeric array")
    return np.asarray(raw, dtype=np.float64)


def _ids(values, n, name):
    values = list(values)
    if len(values) != n or any(not isinstance(v, str) or not v for v in values):
        raise ValueError(name + " must be aligned nonempty string identities")
    return values


def _helmert(count):
    basis = np.zeros((count, max(0, count - 1)), dtype=np.float64)
    for j in range(count - 1):
        scale = math.sqrt((j + 1) * (j + 2))
        basis[: j + 1, j] = 1 / scale
        basis[j + 1, j] = -(j + 1) / scale
    return basis


def _config(kind, rank, prior_kind, dimension, subject_dimension, dtype, ridge, floor_cap):
    if not isinstance(kind, str):
        raise ValueError("kind must be a string")
    if (
        type(dimension) is not int
        or not 1 <= dimension <= 4096
        or type(subject_dimension) is not int
        or not 0 <= subject_dimension <= 4096
    ):
        raise ValueError("invalid item/subject feature dimensions")
    if kind.endswith("_metadata"):
        kind, prior_kind = kind.removesuffix("_metadata"), "metadata"
    kind = {"1pl": "rasch", "2pl": "twopl", "3pl_global": "threepl_global", "3pl_item": "threepl_item"}.get(kind, kind)
    if kind not in KINDS or prior_kind not in ("common", "metadata"):
        raise ValueError("unknown structured recipe/prior")
    if type(rank) is not int or not 1 <= rank <= 8:
        raise ValueError("rank must be an integer in1..8")
    if prior_kind == "metadata" and subject_dimension == 0:
        raise ValueError("metadata prior requires explicit feature columns")
    if dtype not in ("float32", "float64") or type(ridge) not in (float, int) or ridge != 0.01:
        raise ValueError("precision/ridge differs from the fixed recipe")
    if type(floor_cap) not in (int, float) or not math.isfinite(floor_cap) or not 0 <= floor_cap <= 0.25:
        raise ValueError("floor cap must be in[0,.25]")
    return {
        "kind": kind,
        "rank": rank if kind in ("mirt", "caimira") else 1,
        "prior_kind": prior_kind,
        "dimension": dimension,
        "subject_dimension": subject_dimension,
        "dtype": dtype,
        "ridge": 0.01,
        "floor_cap": float(floor_cap),
        "floor_shrinkage": FLOOR_SHRINKAGE,
        "item_floor_shrinkage": ITEM_FLOOR_SHRINKAGE,
        "caimira_l1": CAIMIRA_L1,
        "nuisance_prior": "unit_gaussian_mean_penalty",
        "mirt_gauge": "first_loading_one_bounded_centered_contrasts_training_signed_permutation",
    }


def _shapes(config, subjects=None):
    d, k, z = config["dimension"], config["rank"], config["subject_dimension"]
    shapes = {"intercept": (1,), "difficulty_weight": (d, k if config["kind"] == "caimira" else 1)}
    if config["kind"] in ("twopl", "threepl_global", "threepl_item"):
        shapes.update(discrimination_weight=(d, 1), discrimination_bias=(1,))
    if config["kind"].startswith("threepl"):
        shapes["floor_raw"] = (1,)
    if config["kind"] == "threepl_item":
        shapes["floor_weight"] = (d, 1)
    if config["kind"] == "mirt":
        shapes.update(contrast_weight=(d, k - 1), contrast_bias=(k - 1,))
    if config["kind"] == "caimira":
        shapes.update(relevance_weight=(d, k), relevance_bias=(k,))
    if config["prior_kind"] == "metadata":
        shapes["prior_weight"] = (z, k)
    if subjects is not None:
        shapes["nuisance"] = (max(0, subjects - 1), k)
    return shapes


def _tensor(torch, value, device, dtype):
    return torch.as_tensor(np.array(value, copy=True, order="C"), device=device, dtype=dtype)


def make_problem(
    Xrows,
    y,
    pair_ids,
    subject_ids,
    item_ids,
    subject_features,
    *,
    kind,
    seed,
    rank=2,
    prior_kind="common",
    device="cuda",
    dtype="float32",
    ridge=0.01,
    floor_cap=FLOOR_CAP,
):
    """Prepare a fixed differentiable problem, exposed for independent gradients."""
    x = _array(Xrows, ndim=2, name="item features")
    if not len(x) or not 1 <= x.shape[1] <= 4096:
        raise ValueError("nonempty item features with1..4096 columns required")
    labels = _array(y, ndim=1, name="outcomes")
    if len(labels) != len(x) or not np.isin(labels, [0, 1]).all():
        raise ValueError("strict binary outcomes must match every trial")
    pids = _ids(pair_ids, len(x), "pair IDs")
    sids = _ids(subject_ids, len(x), "subject IDs")
    iids = _ids(item_ids, len(x), "item IDs")
    z = np.empty((len(x), 0)) if subject_features is None else _array(subject_features, ndim=2, name="subject features")
    if len(z) != len(x) or z.shape[1] > 4096:
        raise ValueError("row-aligned subject features required")
    if type(seed) is not int or seed < 0:
        raise ValueError("nonnegative integer seed required")
    config = _config(kind, rank, prior_kind, x.shape[1], z.shape[1], dtype, ridge, floor_cap)
    ibank, sbank, pair_subjects = {}, {}, {}
    for i, (item, subject, pair) in enumerate(zip(iids, sids, pids, strict=True)):
        if item in ibank and not np.array_equal(ibank[item], x[i]):
            raise ValueError("one item identity has inconsistent features")
        if subject in sbank and not np.array_equal(sbank[subject], z[i]):
            raise ValueError("one subject identity has inconsistent semantic features")
        if pair in pair_subjects and pair_subjects[pair] != subject:
            raise ValueError("one pair has multiple subjects")
        ibank[item], sbank[subject], pair_subjects[pair] = x[i], z[i], subject
    items, subjects = sorted(ibank), sorted(sbank)
    ii, si = {v: i for i, v in enumerate(items)}, {v: i for i, v in enumerate(subjects)}
    bx, bz = np.stack([ibank[i] for i in items]), np.stack([sbank[s] for s in subjects])
    present = np.any(bz != 0, axis=1)
    zcenter = bz[present].mean(0) if present.any() else np.zeros(z.shape[1])
    pair_counts = Counter(pids)
    weights = np.asarray([1 / (len(pair_counts) * pair_counts[p]) for p in pids])
    torch, precision = _torch(device, dtype)
    parameters = _parameters(torch, _shapes(config, len(subjects)), seed, device, precision)
    with torch.no_grad():
        parameters["intercept"].zero_()
        for name in ("discrimination_weight", "discrimination_bias", "floor_weight"):
            if name in parameters:
                parameters[name].zero_()
        if "floor_raw" in parameters:
            parameters["floor_raw"].fill_(-4)
        for name in ("contrast_bias", "relevance_bias"):
            if name in parameters:
                parameters[name].zero_()
    initial = {
        key: hashlib.sha256(value.detach().cpu().numpy().astype("<f8").tobytes()).hexdigest()
        for key, value in parameters.items()
    }
    data = {
        "bank": _tensor(torch, bx, device, precision),
        "subject_bank": _tensor(torch, bz, device, precision),
        "subject_center": _tensor(torch, zcenter, device, precision),
        "item_index": _tensor(torch, [ii[v] for v in iids], device, torch.int64),
        "subject_index": _tensor(torch, [si[v] for v in sids], device, torch.int64),
        "labels": _tensor(torch, labels, device, precision),
        "weights": _tensor(torch, weights, device, precision),
        "nuisance_basis": _tensor(torch, _helmert(len(subjects)), device, precision),
    }
    info = {
        "trials": len(x),
        "distinct_items": len(items),
        "distinct_subjects": len(subjects),
        "pair_counts": dict(sorted(pair_counts.items())),
        "weight_sum": float(weights.sum()),
        "item_order_digest": _digest(items),
        "subject_order_digest": _digest(subjects),
        "row_identity_digest": _digest(list(zip(pids, sids, iids, strict=True))),
        "initialization_hashes": initial,
    }
    return parameters, data, config, info


def _item_tensor(parameters, X, bank, config, frozen=None):
    """Return additive offset, ability loadings, floors and diagnostic internals."""
    import torch

    kind, k = config["kind"], config["rank"]
    raw_d = X @ parameters["difficulty_weight"]
    mean_d = (bank @ parameters["difficulty_weight"]).mean(0) if frozen is None else frozen["difficulty"]
    difficulty = raw_d - mean_d
    loadings = X.new_ones((len(X), k))
    mean_a, mean_q = X.new_zeros(()), X.new_zeros((max(0, k - 1),))
    if kind in ("twopl", "threepl_global", "threepl_item"):
        log_a = math.log(2) * torch.tanh(X @ parameters["discrimination_weight"] + parameters["discrimination_bias"])
        mean_a = (
            (
                math.log(2) * torch.tanh(bank @ parameters["discrimination_weight"] + parameters["discrimination_bias"])
            ).mean()
            if frozen is None
            else frozen["discrimination_log"]
        )
        loadings = torch.exp(log_a - mean_a)
    elif kind == "mirt" and k > 1:
        q = torch.tanh(X @ parameters["contrast_weight"] + parameters["contrast_bias"])
        mean_q = (
            torch.tanh(bank @ parameters["contrast_weight"] + parameters["contrast_bias"]).mean(0)
            if frozen is None
            else frozen["contrast"]
        )
        loadings = torch.cat((X.new_ones((len(X), 1)), 0.5 * (q - mean_q)), dim=1)
    elif kind == "caimira":
        scores = X @ parameters["relevance_weight"] + parameters["relevance_bias"]
        loadings = torch.softmax(scores - scores.mean(dim=1, keepdim=True), dim=1)
    offset = parameters["intercept"][0] - (difficulty[:, 0] if kind == "mirt" else (loadings * difficulty).sum(1))
    floors = X.new_zeros((len(X),))
    floor_deviation = X.new_zeros((len(X),))
    if kind.startswith("threepl"):
        if kind == "threepl_item":
            center = bank.mean(0) if frozen is None else frozen["feature"]
            floor_deviation = ((X - center) @ parameters["floor_weight"]).flatten()
        floors = config["floor_cap"] * torch.sigmoid(parameters["floor_raw"][0] + floor_deviation)
    return (
        offset,
        loadings,
        floors,
        {
            "difficulty": difficulty,
            "difficulty_center": mean_d,
            "discrimination_log_center": mean_a,
            "contrast_center": mean_q,
            "floor_deviation": floor_deviation,
        },
    )


def _prior_tensor(parameters, Z, center, config):
    if config["prior_kind"] == "common":
        return Z.new_zeros((len(Z), config["rank"]))
    present = (Z != 0).any(dim=1, keepdim=True)
    return ((Z - center) @ parameters["prior_weight"]) * present


def bernoulli_loss(logits, floors, labels):
    """Actual 3PL likelihood; zero-floor reductions use stable logistic logits."""
    import torch

    base = torch.nn.functional.logsigmoid(logits)
    if bool(torch.any(floors > 0)):
        tiny = torch.finfo(logits.dtype).tiny
        log_one_minus = torch.log1p(-floors)
        log_success = torch.logaddexp(torch.log(floors.clamp(min=tiny)), log_one_minus + base)
        log_failure = log_one_minus + torch.nn.functional.logsigmoid(-logits)
    else:
        log_success, log_failure = base, torch.nn.functional.logsigmoid(-logits)
    return torch.where(labels == 1, -log_success, -log_failure)


def structured_objective(parameters, data, config):
    """Pair-weighted likelihood with explicit nuisance/CAIMIRA/floor penalties."""
    import torch

    bank = data["bank"]
    offsets, loadings, floors, extra = _item_tensor(parameters, bank, bank, config)
    nuisance = data["nuisance_basis"] @ parameters["nuisance"]
    means = _prior_tensor(parameters, data["subject_bank"], data["subject_center"], config)
    ability = means + nuisance
    ix, sx = data["item_index"], data["subject_index"]
    logits = offsets[ix] + (loadings[ix] * ability[sx]).sum(1)
    bce = (data["weights"] * bernoulli_loss(logits, floors[ix], data["labels"])).sum()
    ridge = (
        0.5
        * config["ridge"]
        * sum(p.square().sum() for name, p in parameters.items() if name not in ("nuisance", "floor_raw"))
    )
    prior = 0.5 * nuisance.square().mean()
    special = bce.new_zeros(())
    if config["kind"] == "caimira":
        special = special + config["caimira_l1"] * (
            extra["difficulty"].abs().mean() + (ability + parameters["intercept"][0]).abs().mean()
        )
    if config["kind"].startswith("threepl"):
        if config["floor_cap"]:
            special = special + config["floor_shrinkage"] * (-torch.log1p(-floors / config["floor_cap"])).mean()
        special = special + 0.5 * config["ridge"] * (parameters["floor_raw"][0] + 4).square()
        if config["kind"] == "threepl_item":
            special = special + config["item_floor_shrinkage"] * extra["floor_deviation"].square().mean()
    total = bce + ridge + prior + special
    return total, {"bce": bce, "ridge": ridge, "nuisance_prior": prior, "structured_penalties": special}


def _orientation(parameters, loadings, config):
    k = config["rank"]
    matrix = np.eye(k)
    if config["kind"] == "mirt" and k > 1:
        values = loadings.detach().cpu().numpy()
        weights = parameters["contrast_weight"].detach().cpu().numpy()
        order = sorted(range(k - 1), key=lambda j: (-float(np.var(values[:, j + 1])), j))
        matrix = np.zeros((k, k))
        matrix[0, 0] = 1
        for column, source in enumerate(order, 1):
            anchor = weights[int(np.argmax(np.abs(weights[:, source]))), source]
            matrix[source + 1, column] = 1 if anchor >= 0 else -1
    elif config["kind"] == "caimira" and k > 1:
        values = loadings.detach().cpu().numpy()
        order = sorted(range(k), key=lambda j: (-float(np.var(values[:, j])), j))
        matrix = np.eye(k)[:, order]
    return matrix


class ResidentStructured:
    """Frozen item heads and priors; no training-item or subject lookup path."""

    def __init__(self, state, *, device="cuda", dtype=None):
        if not isinstance(state, dict) or state.get("schema") != SCHEMA:
            raise ValueError("invalid structured state schema")
        if set(state) != {"schema", "config", "parameters", "centers", "orientation"}:
            raise ValueError("unexpected inference state keys, including possible training lookup")
        self.config = dict(state["config"])
        c = self.config
        checked = _config(
            c["kind"],
            c["rank"],
            c["prior_kind"],
            c["dimension"],
            c["subject_dimension"],
            c["dtype"],
            c["ridge"],
            c["floor_cap"],
        )
        if checked != c:
            raise ValueError("structured state recipe differs")
        self.kind, self.rank = c["kind"], c["rank"]
        self.device = device
        self.torch, self.dtype = _torch(device, dtype or c["dtype"])
        shapes = _shapes(c)
        if set(state["parameters"]) != set(shapes):
            raise ValueError("inference parameters contain missing/extra or nuisance terms")
        self.parameters = {}
        for name, shape in shapes.items():
            value = _array(state["parameters"][name], ndim=len(shape), name=name)
            if value.shape != shape:
                raise ValueError("inference parameter shape differs")
            self.parameters[name] = _tensor(self.torch, value, device, self.dtype)
        center_shapes = {
            "difficulty": (c["rank"] if c["kind"] == "caimira" else 1,),
            "feature": (c["dimension"],),
            "subject": (c["subject_dimension"],),
            "contrast": (max(0, c["rank"] - 1),),
            "discrimination_log": (),
        }
        if set(state["centers"]) != set(center_shapes):
            raise ValueError("frozen center names differ")
        self.centers = {}
        for name, shape in center_shapes.items():
            value = _array(state["centers"][name], ndim=len(shape), name="center " + name)
            if value.shape != shape:
                raise ValueError("frozen center shape differs")
            self.centers[name] = _tensor(self.torch, value, device, self.dtype)
        orientation = _array(state["orientation"], ndim=2, name="orientation")
        if orientation.shape != (self.rank, self.rank) or not np.allclose(
            orientation.T @ orientation, np.eye(self.rank), atol=1e-12
        ):
            raise ValueError("orientation must be an orthogonal declared coordinate transform")
        if self.kind == "caimira" and (not np.isin(orientation, (0, 1)).all() or not np.all(orientation.sum(0) == 1)):
            raise ValueError("CAIMIRA orientation must preserve simplex coordinates")
        self.orientation = _tensor(self.torch, orientation, device, self.dtype)
        self._state = json.loads(json.dumps(state, allow_nan=False))

    @classmethod
    def from_state(cls, state, *, device="cuda", dtype=None):
        return cls(state, device=device, dtype=dtype)

    def export(self):
        return json.loads(json.dumps(self._state, allow_nan=False))

    def item_parameters_tensor(self, X):
        values = _array(X, ndim=2, name="query item features")
        if values.shape[1] != self.config["dimension"]:
            raise ValueError("unseen item feature dimension differs")
        with self.torch.inference_mode():
            tensor = _tensor(self.torch, values, self.device, self.dtype)
            offset, loading, floor, _ = _item_tensor(self.parameters, tensor, None, self.config, self.centers)
            return offset, loading @ self.orientation, floor

    def item_parameters(self, X):
        return tuple(value.cpu().numpy() for value in self.item_parameters_tensor(X))

    def prior_tensor(self, subject_features):
        values = _array(subject_features, ndim=2, name="query subject metadata")
        if values.shape[1] != self.config["subject_dimension"]:
            raise ValueError("unseen subject feature dimension differs")
        with self.torch.inference_mode():
            return (
                _prior_tensor(
                    self.parameters,
                    _tensor(self.torch, values, self.device, self.dtype),
                    self.centers["subject"],
                    self.config,
                )
                @ self.orientation
            )

    def prior(self, subject_features):
        return self.prior_tensor(subject_features).cpu().numpy()

    def prior_covariance(self):
        return np.eye(self.rank, dtype=np.float64)


def fit_structured(
    Xrows,
    y,
    pair_ids,
    subject_ids,
    item_ids,
    subject_features=None,
    *,
    kind,
    seed,
    rank=2,
    prior_kind="common",
    device="cuda",
    dtype="float32",
    steps=300,
    max_seconds=1200,
    ridge=0.01,
    floor_cap=FLOOR_CAP,
):
    """Fit exactly the requested update count; callers freeze scientific steps300."""
    if type(steps) is not int or not 1 <= steps <= 300 or not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("bounded1..300 updates and positive fit deadline required")
    p, data, config, info = make_problem(
        Xrows,
        y,
        pair_ids,
        subject_ids,
        item_ids,
        subject_features,
        kind=kind,
        seed=seed,
        rank=rank,
        prior_kind=prior_kind,
        device=device,
        dtype=dtype,
        ridge=ridge,
        floor_cap=floor_cap,
    )
    torch, _ = _torch(device, dtype)
    diagnostic = _optimize(torch, p, lambda: structured_objective(p, data, config), steps, max_seconds)
    with torch.no_grad():
        bank = data["bank"]
        offset, loading, floor, extra = _item_tensor(p, bank, bank, config)
        orientation = _orientation(p, loading, config)
        state = {
            "schema": SCHEMA,
            "config": config,
            "parameters": {name: value.detach().cpu().tolist() for name, value in p.items() if name != "nuisance"},
            "centers": {
                "difficulty": extra["difficulty_center"].cpu().tolist(),
                "feature": bank.mean(0).cpu().tolist(),
                "subject": data["subject_center"].cpu().tolist(),
                "contrast": extra["contrast_center"].cpu().tolist(),
                "discrimination_log": extra["discrimination_log_center"].cpu().tolist(),
            },
            "orientation": orientation.tolist(),
        }
        model = ResidentStructured(state, device=device)
        nuisance = data["nuisance_basis"] @ p["nuisance"]
        means = _prior_tensor(p, data["subject_bank"], data["subject_center"], config)
        ix, sx = data["item_index"], data["subject_index"]
        eta = offset[ix] + (loading[ix] * (means + nuisance)[sx]).sum(1)
        base_probability = torch.sigmoid(eta)
        probability = floor[ix] + (1 - floor[ix]) * base_probability
        slope = (1 - floor[ix]) * base_probability * (1 - base_probability)
        fisher = slope.square() / (probability * (1 - probability)).clamp(min=1e-12)
        information = []
        for subject in range(info["distinct_subjects"]):
            keep = sx == subject
            local = loading[ix][keep]
            gram = local.T @ (fisher[keep, None] * local)
            eigen = torch.linalg.eigvalsh(gram).cpu().tolist()
            information.append(
                {
                    "likelihood_eigenvalues": eigen,
                    "prior_precision_fraction": [1 / (1 + max(0, value)) for value in eigen],
                }
            )
        receipt = {
            "schema": "proxy-structured-fit/v1",
            **info,
            "kind": config["kind"],
            "rank": config["rank"],
            "prior_kind": config["prior_kind"],
            "seed": seed,
            "device": device,
            "dtype": dtype,
            "steps": steps,
            "optimizer": "Adam",
            "learning_rate": 0.01,
            "fused": False,
            "foreach": False,
            "diagnostics": diagnostic,
            "state_sha256": _digest(state),
            "difficulty_center": state["centers"]["difficulty"],
            "difficulty_training_mean": extra["difficulty"].mean(0).cpu().tolist(),
            "loading_singular_values": torch.linalg.svdvals(loading).cpu().tolist(),
            "information_by_training_subject": information,
            "information_scope": "Unweighted training-trial Fisher; precision fractions compare unit deployment prior, not a training-posterior identification claim",
            "prior_dependence": {
                "runtime_prior_covariance": "identity",
                "metadata_mean_square": float(means.square().mean().cpu()),
                "nuisance_mean_square": float(nuisance.square().mean().cpu()),
            },
            "discrimination_geometric_mean": float(torch.exp(torch.log(loading).mean()).cpu())
            if config["kind"] in ("twopl", "threepl_global", "threepl_item")
            else None,
            "floor_summary": {
                "minimum": float(floor.min().cpu()),
                "maximum": float(floor.max().cpu()),
                "mean": float(floor.mean().cpu()),
                "std": float(floor.std(unbiased=False).cpu()),
            },
            "training_only": {
                "nuisance_abilities": nuisance.cpu().tolist(),
                "subject_order_digest": info["subject_order_digest"],
            },
            "export_has_training_lookup": False,
            "fallback_calls": 0,
            "identification_scope": "Anchored and regularized coordinates; finite/posterior-unique does not imply label-supported identification.",
        }
    return model, receipt


def qualification_gate(
    dtype, *, probability_difference, initialization, convergence, finite, failure_dispositions, objectives, gradients
):
    """Prospective precision-specific backend gate; FP32 trajectories are diagnostic."""
    if dtype not in ("float64", "float32"):
        raise ValueError("unknown qualification precision")
    tolerance = 1e-8 if dtype == "float64" else 1e-5
    return bool(
        math.isfinite(probability_difference)
        and probability_difference <= tolerance
        and initialization
        and convergence
        and finite
        and failure_dispositions
        and (dtype == "float32" or (objectives and gradients))
    )


def numerical_preflight(*, device="cuda"):
    """Explicit synthetic300-step CPU/device qualification, never called on import."""
    import gc
    import time

    if device not in ("cuda", "cuda:0"):
        raise ValueError("backend qualification requires an explicit CUDA target")
    rng = np.random.default_rng(20261030)
    item = rng.normal(0, 0.2, (8, 4))
    subject = np.asarray([[0.2, -0.1, 0.3], [0.0, 0.0, 0.0], [-0.2, 0.4, -0.1]])
    x, z, y, pairs, subjects, items = [], [], [], [], [], []
    for s in range(3):
        for i in range(8):
            x.append(item[i])
            z.append(subject[s])
            y.append(int((s + 3 * i) % 2))
            pairs.append("pair" + str(s))
            subjects.append("subject" + str(s))
            items.append("item" + str(i))
    for values in (x, z, y, pairs, subjects, items):
        values.append(values[0])
    recipes = (
        ("rasch", "common"),
        ("twopl", "common"),
        ("threepl_global", "common"),
        ("threepl_item", "common"),
        ("mirt", "common"),
        ("caimira", "common"),
        ("rasch", "metadata"),
        ("caimira", "metadata"),
    )
    results = {
        "schema": "proxy-structured-numerical-preflight/v1",
        "target_device": device,
        "synthetic_only": True,
        "seed": 17,
        "steps": 300,
    }
    for dtype in ("float64", "float32"):
        cases = []
        for kind, prior in recipes:
            records, probabilities, timings = {}, {}, {}
            for target in ("cpu", device):
                started = time.monotonic()
                model, receipt = fit_structured(
                    x,
                    y,
                    pairs,
                    subjects,
                    items,
                    z,
                    kind=kind,
                    prior_kind=prior,
                    seed=17,
                    rank=2,
                    device=target,
                    dtype=dtype,
                    steps=300,
                )
                offset, loading, floor = model.item_parameters(x)
                mean = model.prior(z)
                logits = offset + np.sum(loading * mean, axis=1)
                probabilities[target] = floor + (1 - floor) / (1 + np.exp(-logits))
                records[target] = receipt
                timings[target] = time.monotonic() - started
                del model
                gc.collect()
            cpu, actual = records["cpu"], records[device]
            difference = float(np.max(np.abs(probabilities["cpu"] - probabilities[device])))
            trajectories = {}
            for label in ("objective", "gradient_inf"):
                left = np.asarray([row[label] for row in cpu["diagnostics"]["history"]])
                right = np.asarray([row[label] for row in actual["diagnostics"]["history"]])
                trajectories[label] = {
                    "pass": bool(np.allclose(left, right, atol=1e-8, rtol=1e-6)),
                    "maximum_absolute_difference": float(np.max(np.abs(left - right))),
                    "finite_pass": bool(np.isfinite(left).all() and np.isfinite(right).all()),
                }
            initialization = cpu["initialization_hashes"] == actual["initialization_hashes"]
            convergence = cpu["diagnostics"]["convergence"] == actual["diagnostics"]["convergence"]
            finite = all(np.isfinite(v).all() for v in probabilities.values()) and all(
                v["finite_pass"] for v in trajectories.values()
            )
            failure_dispositions = cpu["fallback_calls"] == actual["fallback_calls"] == 0
            passed = qualification_gate(
                dtype,
                probability_difference=difference,
                initialization=initialization,
                convergence=convergence,
                finite=finite,
                failure_dispositions=failure_dispositions,
                objectives=trajectories["objective"]["pass"],
                gradients=trajectories["gradient_inf"]["pass"],
            )
            cases.append(
                {
                    "kind": kind,
                    "prior_kind": prior,
                    "status": "PASS" if passed else "FAIL",
                    "max_probability_difference": difference,
                    "initialization_hashes_equal": initialization,
                    "trajectory_gate_applied": dtype == "float64",
                    "finite_pass": bool(finite),
                    "failure_dispositions_pass": failure_dispositions,
                    "convergence_pass": convergence,
                    "objectives_pass": trajectories["objective"]["pass"],
                    "gradients_pass": trajectories["gradient_inf"]["pass"],
                    "trajectories": trajectories,
                    "timings_seconds": timings,
                    "convergence": {k: v["diagnostics"]["convergence"] for k, v in records.items()},
                }
            )
        results[dtype] = {
            "status": "PASS" if all(c["status"] == "PASS" for c in cases) else "FAIL",
            "max_probability_difference": max(c["max_probability_difference"] for c in cases),
            "objectives_pass": all(c["objectives_pass"] for c in cases),
            "gradients_pass": all(c["gradients_pass"] for c in cases),
            "trajectory_gate_applied": dtype == "float64",
            "finite_pass": all(c["finite_pass"] for c in cases),
            "failure_dispositions_pass": all(c["failure_dispositions_pass"] for c in cases),
            "initialization_hashes_equal": all(c["initialization_hashes_equal"] for c in cases),
            "convergence_pass": all(c["convergence_pass"] for c in cases),
            "cases": cases,
        }
    results["status"] = "PASS" if all(results[d]["status"] == "PASS" for d in ("float64", "float32")) else "FAIL"
    return results
