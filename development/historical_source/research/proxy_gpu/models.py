"""Versioned dense CPU/CUDA math for prospective technique ablations.

Frozen CPU modules are never modified. This backend uses canonical CPU random
initialization, explicit float32/float64 arithmetic and resident inference
parameters. Callers own fold admission, legal evidence and external deadlines.
Scalar Gaussian MAP/state quadrature retain the audited analytic CPU solver;
query prediction and dense pooling are batched on the requested device.
"""

from __future__ import annotations

import math
import os
import time
from collections import Counter

import numpy as np

from research.roadmap import numerics

HEAD_SCHEMA = "proxy-gpu-head/v1"
RESIDUAL_SCHEMA = "proxy-gpu-residual/v1"
MIXTURE_SCHEMA = "proxy-gpu-mixture/v1"
RIDGE = 0.01
RANK = 8
WIDTH = 16


def _contains_boolean(value):
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray):
        return value.dtype.kind == "b"
    return isinstance(value, (list, tuple)) and any(_contains_boolean(part) for part in value)


def _array(value, *, ndim, name, allow_empty=True):
    if _contains_boolean(value):
        raise ValueError(name + " must not contain Boolean values")
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or array.ndim != ndim:
        raise ValueError(name + " requires a numeric array of the declared dimension")
    result = np.asarray(array, dtype=np.float64)
    if not np.isfinite(result).all() or (not allow_empty and len(result) == 0):
        raise ValueError(name + " requires finite nonempty observations")
    if result.size > 125_000_000:
        raise ValueError(name + " exceeds the bounded element count")
    return result


def _matrix(value, name="features", allow_empty=True):
    result = _array(value, ndim=2, name=name, allow_empty=allow_empty)
    if result.shape[1] > 16384 or len(result) > 50000:
        raise ValueError(name + " exceeds the bounded matrix contract")
    return result


def _labels(value, count):
    result = _array(value, ndim=1, name="labels")
    if result.shape != (count,) or not np.isin(result, (0, 1)).all():
        raise ValueError("one exact numeric binary label is required per observation")
    return result


def _vector(value, count=None, name="logits", allow_empty=True):
    result = _array(value, ndim=1, name=name, allow_empty=allow_empty)
    if count is not None and len(result) != count:
        raise ValueError(name + " has a different observation count")
    return result


def _pair_ids(pair_ids, count):
    if (
        not isinstance(pair_ids, (list, tuple))
        or len(pair_ids) != count
        or any(type(pair) is not str or not pair for pair in pair_ids)
    ):
        raise ValueError("complete nonempty pair identities are required")
    return Counter(pair_ids)


def _settings(seed, steps, max_seconds):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an unsigned 32-bit integer")
    if type(steps) is not int or not 1 <= steps <= 300:
        raise ValueError("1..300 optimizer updates must be declared")
    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 0 < max_seconds <= 1200:
        raise ValueError("cooperative fit deadline must be positive and at most 1200 seconds")


def _torch(device, dtype):
    if device not in {"cpu", "cuda"} or dtype not in {"float32", "float64"}:
        raise ValueError("device is cpu/cuda and dtype is float32/float64")
    import torch

    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace not in (None, ":4096:8"):
        raise RuntimeError("conflicting deterministic cuBLAS workspace configuration")
    if device == "cuda" and torch.cuda.is_initialized() and workspace != ":4096:8":
        raise RuntimeError("CUDA initialized before deterministic cuBLAS configuration")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; no CPU fallback")
    torch.use_deterministic_algorithms(True)
    torch.backends.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    torch.backends.cudnn.rnn.fp32_precision = "ieee"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return torch, getattr(torch, dtype)


def _tensor(torch, values, device, dtype):
    return torch.as_tensor(values, dtype=dtype, device=device)


def _parameters(torch, shapes, seed, device, dtype):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return {
        name: torch.nn.Parameter(
            (torch.randn(shape, generator=generator, dtype=torch.float64) * 0.01).to(device=device, dtype=dtype)
        )
        for name, shape in shapes.items()
    }


def _optimize(torch, parameters, objective, steps, max_seconds):
    optimizer = torch.optim.Adam(
        list(parameters.values()),
        lr=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        foreach=False,
        fused=False,
    )
    started = time.monotonic()
    history = []
    for step in range(steps + 1):
        if time.monotonic() - started > max_seconds:
            raise TimeoutError("prospective dense fit exceeded its cooperative deadline")
        optimizer.zero_grad(set_to_none=True)
        loss, pieces = objective()
        loss.backward()
        row = {
            "step": step,
            "objective": float(loss.detach().cpu()),
            "gradient_inf": max(float(p.grad.abs().max().detach().cpu()) for p in parameters.values() if p.numel()),
            **{key: float(value.detach().cpu()) for key, value in pieces.items()},
        }
        if not all(math.isfinite(value) for value in row.values()):
            raise ArithmeticError("nonfinite dense training state")
        history.append(row)
        if step < steps:
            optimizer.step()
    previous = history[max(0, steps - 20)]
    relative = abs(history[-1]["objective"] - previous["objective"]) / max(1.0, abs(previous["objective"]))
    return {
        "history": history,
        "relative_objective_change": relative,
        "convergence": (
            "CONVERGED" if history[-1]["gradient_inf"] <= 1e-4 and relative <= 1e-6 else "BOUNDED_UNCONVERGED"
        ),
        "elapsed_seconds": time.monotonic() - started,
        "max_seconds": max_seconds,
        "deadline_semantics": "cooperative_between_updates_external_hard_limit_required",
        "fallback_calls": 0,
    }


def _training(seed, steps, device, dtype, diagnostic):
    return {
        "seed": seed,
        "steps": steps,
        "device": device,
        "dtype": dtype,
        "optimizer": "Adam",
        "learning_rate": 0.01,
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "ridge": RIDGE,
        "foreach": False,
        "fused": False,
        "initialization": "cpu_float64_generator_then_copy_and_cast",
        "convergence": diagnostic["convergence"],
    }


def _dense_logits(parameters, x, subject, kind):
    logits = parameters["bias"][0] + x @ parameters["item"]
    if kind != "item":
        logits = logits + subject @ parameters["subject"]
    if kind == "bilinear":
        logits = logits + ((x @ parameters["left"]) * (subject @ parameters["right"])).sum(dim=1)
    return logits


def fit_head(
    x,
    y,
    pair_ids,
    *,
    seed,
    kind="item",
    subject_features=None,
    device="cuda",
    dtype="float32",
    steps=300,
    max_seconds=1200,
):
    """Fit item, additive subject+item or additive rank-eight bilinear logits.

    Response weights sum equally within each pair. Centers are training-only,
    using the same pair weights. All parameters, including biases, receive ridge.
    The returned JSON object contains every fitted parameter and diagnostic.
    """
    _settings(seed, steps, max_seconds)
    if kind not in {"item", "additive", "bilinear"}:
        raise ValueError("unknown dense head kind")
    values = _matrix(x, allow_empty=False)
    labels = _labels(y, len(values))
    counts = _pair_ids(pair_ids, len(values))
    if kind == "item":
        if subject_features is not None:
            raise ValueError("item-only head must not receive subject features")
        subjects = np.empty((len(values), 0), dtype=np.float64)
    else:
        subjects = _matrix(subject_features, "subject features", allow_empty=False)
        if len(subjects) != len(values) or not subjects.shape[1] or not values.shape[1]:
            raise ValueError("subject and item feature rows must match with nonempty dimensions")
    torch, precision = _torch(device, dtype)
    weight = _tensor(torch, [1 / (len(counts) * counts[p]) for p in pair_ids], device, precision)
    tensor = _tensor(torch, values, device, precision)
    subject = _tensor(torch, subjects, device, precision)
    center = (tensor * weight[:, None]).sum(dim=0)
    subject_center = (subject * weight[:, None]).sum(dim=0)
    tensor, subject = tensor - center, subject - subject_center
    target = _tensor(torch, labels, device, precision)
    shapes = {"bias": (1,), "item": (values.shape[1],)}
    if kind != "item":
        shapes["subject"] = (subjects.shape[1],)
    if kind == "bilinear":
        shapes.update(left=(values.shape[1], RANK), right=(subjects.shape[1], RANK))
    parameters = _parameters(torch, shapes, seed, device, precision)

    def objective():
        logits = _dense_logits(parameters, tensor, subject, kind)
        bce = (weight * torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")).sum()
        penalty = 0.5 * RIDGE * sum(value.square().sum() for value in parameters.values())
        return bce + penalty, {"bce": bce, "ridge": penalty}

    diagnostic = _optimize(torch, parameters, objective, steps, max_seconds)
    diagnostic.update(
        pair_count=len(counts),
        response_count=len(values),
        weight_sum=float(weight.sum().detach().cpu()),
        pair_counts=dict(sorted(counts.items())),
    )
    return {
        "schema": HEAD_SCHEMA,
        "kind": kind,
        "dimension": values.shape[1],
        "subject_dimension": subjects.shape[1],
        "rank": RANK if kind == "bilinear" else 0,
        "center": center.detach().cpu().tolist(),
        "subject_center": subject_center.detach().cpu().tolist(),
        "parameters": {key: value.detach().cpu().tolist() for key, value in parameters.items()},
        "training": {
            **_training(seed, steps, device, dtype, diagnostic),
            "weighting": "equal_pair_then_equal_response_within_pair",
            "centering": "training_only_pair_weighted",
        },
        "diagnostics": diagnostic,
    }


class ResidentHead:
    """Retain dense parameters on one device; evaluate complete query batches."""

    def __init__(self, head, device="cuda"):
        if not isinstance(head, dict) or head.get("schema") != HEAD_SCHEMA:
            raise ValueError("unsupported resident dense head")
        self.kind = head.get("kind")
        if self.kind not in {"item", "additive", "bilinear"}:
            raise ValueError("unsupported resident head kind")
        self.dimension = head["dimension"]
        self.subject_dimension = head["subject_dimension"]
        if any(type(d) is not int or not 0 <= d <= 16384 for d in (self.dimension, self.subject_dimension)):
            raise ValueError("invalid resident head dimensions")
        if (self.kind == "item" and self.subject_dimension != 0) or (
            self.kind != "item" and (not self.dimension or not self.subject_dimension)
        ):
            raise ValueError("head kind and dimensions differ")
        if head.get("rank") != (RANK if self.kind == "bilinear" else 0):
            raise ValueError("invalid resident head rank")
        self.torch, self.dtype = _torch(device, head["training"]["dtype"])
        self.device = device
        self.center = _tensor(self.torch, _vector(head["center"], self.dimension, "center"), device, self.dtype)
        self.subject_center = _tensor(
            self.torch, _vector(head["subject_center"], self.subject_dimension, "subject center"), device, self.dtype
        )
        shapes = {"bias": (1,), "item": (self.dimension,)}
        if self.kind != "item":
            shapes["subject"] = (self.subject_dimension,)
        if self.kind == "bilinear":
            shapes.update(left=(self.dimension, RANK), right=(self.subject_dimension, RANK))
        if set(head["parameters"]) != set(shapes):
            raise ValueError("head parameter names differ")
        self.parameters = {}
        for name, shape in shapes.items():
            value = np.asarray(head["parameters"][name])
            if value.shape != shape or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
                raise ValueError("head parameter shape/value is invalid")
            self.parameters[name] = _tensor(self.torch, value, device, self.dtype)

    def _logits(self, x, subject_features):
        values = _matrix(x)
        if values.shape[1] != self.dimension:
            raise ValueError("query feature dimension differs")
        if self.kind == "item":
            if subject_features is not None:
                raise ValueError("item head must not receive subject features")
            subjects = np.empty((len(values), 0))
        else:
            subjects = _matrix(subject_features, "query subject features")
            if subjects.shape != (len(values), self.subject_dimension):
                raise ValueError("query subject feature dimension differs")
        tensor = _tensor(self.torch, values, self.device, self.dtype) - self.center
        subject = _tensor(self.torch, subjects, self.device, self.dtype) - self.subject_center
        result = _dense_logits(self.parameters, tensor, subject, self.kind)
        if not bool(self.torch.isfinite(result).all()):
            raise ArithmeticError("nonfinite resident head logits")
        return result

    def logits(self, x, subject_features=None):
        with self.torch.inference_mode():
            return self._logits(x, subject_features).cpu().numpy()

    def predict(self, x, subject_features=None):
        with self.torch.inference_mode():
            return self.torch.sigmoid(self._logits(x, subject_features)).cpu().numpy()


def _episode_weights(episodes):
    if not isinstance(episodes, (list, tuple)) or not episodes:
        raise ValueError("at least one training episode is required")
    weights = []
    for episode in episodes:
        if not isinstance(episode, dict):
            raise ValueError("episodes must be explicit dictionaries")
        weight = episode.get("weight")
        if type(weight) not in (int, float) or not math.isfinite(weight) or weight <= 0:
            raise ValueError("episode weights must be positive finite numbers")
        if type(episode.get("pair_id")) is not str or not episode["pair_id"]:
            raise ValueError("episode pair identity is required")
        weights.append(float(weight))
    if not math.isclose(math.fsum(weights), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("episode weights must sum to one")
    return weights


def _summary(torch, support, labels, kind):
    count = len(labels)
    totals = torch.stack(
        (
            labels.new_tensor(math.log1p(count)),
            labels.sum() / max(1, count),
        )
    )
    if kind == "counts":
        return totals
    success = labels == 1
    failure = labels == 0
    zero = support.new_zeros(support.shape[1])
    good = support[success].mean(dim=0) if bool(success.any()) else zero
    bad = support[failure].mean(dim=0) if bool(failure.any()) else zero
    return torch.cat((totals, good, bad))


def _support(features, labels, dimension=None):
    # Empty support must retain its feature dimension, e.g. np.empty((0,D)).
    values = _matrix(features, "support features")
    if dimension is not None and values.shape[1] != dimension:
        raise ValueError("support feature dimension differs")
    y = _labels(labels, len(values))
    if len(y) > 31:
        raise ValueError("at most 31 purchased support observations are permitted")
    return values, y


def _residual_logits(parameters, summary):
    return (summary @ parameters["weights_in"] + parameters["bias_in"]).tanh() @ parameters["weights_out"] + parameters[
        "bias_out"
    ][0]


def fit_residual(
    episodes,
    *,
    seed,
    kind="counts",
    device="cuda",
    dtype="float32",
    steps=300,
    max_seconds=1200,
):
    """Fit a width-16 tanh scalar correction to a frozen base head.

    Each episode supplies base_logits, y, support_features, support_labels,
    pair_id and normalized weight. Query outcomes train the correction only;
    inference accepts no query labels. No query features enter this summary.
    """
    _settings(seed, steps, max_seconds)
    if kind not in {"counts", "features"}:
        raise ValueError("residual kind is counts/features")
    weights = _episode_weights(episodes)
    torch, precision = _torch(device, dtype)
    dimension = None
    packed = []
    for episode, weight in zip(episodes, weights, strict=True):
        values, labels = _support(episode["support_features"], episode["support_labels"], dimension)
        dimension = values.shape[1]
        base = _vector(episode["base_logits"], name="base logits", allow_empty=False)
        target = _labels(episode["y"], len(base))
        summary = _summary(
            torch, _tensor(torch, values, device, precision), _tensor(torch, labels, device, precision), kind
        )
        packed.append(
            (summary, _tensor(torch, base, device, precision), _tensor(torch, target, device, precision), weight)
        )
    width = 2 if kind == "counts" else 2 + 2 * dimension
    parameters = _parameters(
        torch,
        {"weights_in": (width, WIDTH), "bias_in": (WIDTH,), "weights_out": (WIDTH,), "bias_out": (1,)},
        seed,
        device,
        precision,
    )

    def objective():
        loss = packed[0][1].new_zeros(())
        for summary, base, target, weight in packed:
            logits = base + _residual_logits(parameters, summary)
            loss = loss + weight * torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        penalty = 0.5 * RIDGE * sum(value.square().sum() for value in parameters.values())
        return loss + penalty, {"bce": loss, "ridge": penalty}

    diagnostic = _optimize(torch, parameters, objective, steps, max_seconds)
    diagnostic.update(episode_count=len(episodes), pair_count=len({e["pair_id"] for e in episodes}))
    return {
        "schema": RESIDUAL_SCHEMA,
        "kind": kind,
        "dimension": dimension,
        "summary_dimension": width,
        "width": WIDTH,
        "summary": "log1p_count_success_rate_then_success_and_failure_means_empty_zero",
        "parameters": {key: value.detach().cpu().tolist() for key, value in parameters.items()},
        "training": _training(seed, steps, device, dtype, diagnostic),
        "diagnostics": diagnostic,
    }


class ResidentResidual:
    """A resident support-summary correction; neither attention nor query labels."""

    def __init__(self, head, device="cuda"):
        if not isinstance(head, dict) or head.get("schema") != RESIDUAL_SCHEMA:
            raise ValueError("unsupported resident residual")
        self.kind = head["kind"]
        if self.kind not in {"counts", "features"} or head["width"] != WIDTH:
            raise ValueError("unsupported residual kind/width")
        self.dimension = head["dimension"]
        if type(self.dimension) is not int or not 0 <= self.dimension <= 16384:
            raise ValueError("invalid residual feature dimension")
        width = 2 if self.kind == "counts" else 2 + 2 * self.dimension
        if head["summary_dimension"] != width:
            raise ValueError("residual summary dimension differs")
        self.torch, self.dtype = _torch(device, head["training"]["dtype"])
        self.device = device
        shapes = {
            "weights_in": (width, WIDTH),
            "bias_in": (WIDTH,),
            "weights_out": (WIDTH,),
            "bias_out": (1,),
        }
        if set(head["parameters"]) != set(shapes):
            raise ValueError("residual parameter names differ")
        self.parameters = {}
        for name, shape in shapes.items():
            value = np.asarray(head["parameters"][name])
            if value.shape != shape or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
                raise ValueError("invalid residual parameter")
            self.parameters[name] = _tensor(self.torch, value, device, self.dtype)

    def _logits(self, base_logits, support_features, support_labels):
        base = _vector(base_logits, name="base logits", allow_empty=False)
        features, labels = _support(support_features, support_labels, self.dimension)
        tensor = _tensor(self.torch, base, self.device, self.dtype)
        summary = _summary(
            self.torch,
            _tensor(self.torch, features, self.device, self.dtype),
            _tensor(self.torch, labels, self.device, self.dtype),
            self.kind,
        )
        logits = tensor + _residual_logits(self.parameters, summary)
        if not bool(self.torch.isfinite(logits).all()):
            raise ArithmeticError("nonfinite residual prediction")
        return logits

    def logits(self, base_logits, support_features, support_labels):
        with self.torch.inference_mode():
            return self._logits(base_logits, support_features, support_labels).cpu().numpy()

    def predict(self, base_logits, support_features, support_labels):
        with self.torch.inference_mode():
            return self.torch.sigmoid(self._logits(base_logits, support_features, support_labels)).cpu().numpy()


def pooling_probabilities(query_features, support_features, labels, *, tau=4.0, device="cuda", dtype="float32"):
    """Dense cosine pooling with PM8 shrinkage, matching the fixed-kernel rule."""
    query = _matrix(query_features, "query features", allow_empty=False)
    support, y = _support(support_features, labels, query.shape[1])
    if type(tau) not in (int, float) or not math.isfinite(tau) or not 0 <= tau <= 8:
        raise ValueError("pooling tau must be finite in [0,8]")
    torch, precision = _torch(device, dtype)
    with torch.inference_mode():
        q, s = _tensor(torch, query, device, precision), _tensor(torch, support, device, precision)
        target = _tensor(torch, y, device, precision)
        if not len(y):
            return torch.full((len(q),), 0.5, dtype=precision, device=device).cpu().numpy()
        if tau == 0 or bool(torch.all(target == target[0])):
            return (
                torch.full((len(q),), (4 + float(np.sum(y))) / (8 + len(y)), dtype=precision, device=device)
                .cpu()
                .numpy()
            )
        q_norm = torch.linalg.vector_norm(q, dim=1)
        s_norm = torch.linalg.vector_norm(s, dim=1)
        q = q / torch.where(q_norm > 0, q_norm, 1)[:, None]
        s = s / torch.where(s_norm > 0, s_norm, 1)[:, None]
        cosine = (q @ s.T).clamp(-1, 1)
        both_zero = (q_norm == 0)[:, None] & (s_norm == 0)[None, :]
        cosine = torch.where(both_zero, 1, cosine)
        weight = torch.softmax(tau * cosine, dim=1)
        estimate = weight @ target
        return ((4 + len(y) * estimate) / (8 + len(y))).cpu().numpy()


def fit_mixture(episodes, *, device="cuda", dtype="float64"):
    """Analytic clamped Brier-optimal mixture on legal outer-training episodes.

    Episodes contain base_probabilities, alternate_probabilities, y, pair_id,
    weight. Each query trial retains weight/number_of_queries. A zero contrast
    denominator retains the base endpoint rather than choosing an arbitrary tie.
    """
    weights = _episode_weights(episodes)
    torch, precision = _torch(device, dtype)
    numerator = torch.zeros((), dtype=precision, device=device)
    denominator = torch.zeros_like(numerator)
    constant = torch.zeros_like(numerator)
    count = 0
    with torch.inference_mode():
        for episode, weight in zip(episodes, weights, strict=True):
            base = _vector(episode["base_probabilities"], name="base probabilities", allow_empty=False)
            alternate = _vector(episode["alternate_probabilities"], len(base), "alternate probabilities")
            if np.any((base < 0) | (base > 1) | (alternate < 0) | (alternate > 1)):
                raise ValueError("mixture endpoints must be probabilities")
            target = _labels(episode["y"], len(base))
            b, a, y = (_tensor(torch, value, device, precision) for value in (base, alternate, target))
            residual, difference = b - y, a - b
            numerator = numerator + weight * (residual * difference).mean()
            denominator = denominator + weight * difference.square().mean()
            constant = constant + weight * residual.square().mean()
            count += len(base)
        value = (-numerator / denominator).clamp(0, 1) if bool(denominator > 0) else torch.zeros_like(denominator)
        objective = constant + 2 * value * numerator + value.square() * denominator
        result = {
            "schema": MIXTURE_SCHEMA,
            "weight": float(value.cpu()),
            "formula": "(1-weight)*base+weight*alternate",
            "objective": "episode_weighted_response_mean_brier",
            "brier": float(objective.cpu()),
            "base_brier": float(constant.cpu()),
            "denominator": float(denominator.cpu()),
            "numerator": float(numerator.cpu()),
            "episode_count": len(episodes),
            "response_count": count,
            "device": device,
            "dtype": dtype,
            "fallback_calls": 0,
        }
    if not all(math.isfinite(result[key]) for key in ("weight", "brier", "base_brier", "denominator", "numerator")):
        raise ArithmeticError("nonfinite fitted mixture")
    return result


def adapt_probabilities(
    query_logits,
    support_logits,
    support_labels,
    *,
    update="map",
    device="cuda",
    dtype="float64",
):
    """Audited N(0,1) scalar support-state solve plus batched device queries.

    The CPU analytic solver performs bounded Newton and GH21/GH41 state checks.
    GPU posterior probabilities independently retain the GH21/GH41 query check.
    No numerical failure silently falls back to a constant or unadapted answer.
    """
    query = _vector(query_logits, name="query logits", allow_empty=False)
    support = _vector(support_logits, name="support logits")
    labels = _labels(support_labels, len(support))
    if len(support) > 31 or update not in {"none", "map", "posterior"}:
        raise ValueError("at most 31 support observations and a declared update are required")
    torch, precision = _torch(device, dtype)
    diagnostics = {
        "update": update,
        "state_device": "cpu_analytic",
        "query_device": device,
        "dtype": dtype,
        "support_count": len(support),
        "query_count": len(query),
        "fallback_calls": 0,
    }
    with torch.inference_mode():
        tensor = _tensor(torch, query, device, precision)
        if update == "none":
            probability = torch.sigmoid(tensor)
        else:
            state_input = numerics.prepare_support(
                support.tolist(), [(1.0,)] * len(support), [int(y) for y in labels], (0.0,)
            )
            mode = numerics.solve_map(state_input)
            if mode.status != "OK":
                raise ArithmeticError("MAP failure: " + str(mode.reason))
            diagnostics.update(map_mode=mode.mode[0], map_iterations=mode.iterations, map_status=mode.status)
            if update == "map":
                probability = torch.sigmoid(tensor + mode.mode[0])
            else:
                state = numerics.prepare_posterior(state_input, mode)
                if state.status != "OK":
                    raise ArithmeticError("posterior failure: " + str(state.reason))
                probabilities = []
                for grid in (state.production, state.check):
                    points = _tensor(torch, [point[0] for point in grid.points], device, precision)
                    weights = _tensor(torch, grid.normalized_weights, device, precision)
                    probabilities.append((torch.sigmoid(tensor[:, None] + points[None, :]) * weights).sum(dim=1))
                discrepancy = float((probabilities[0] - probabilities[1]).abs().max().cpu())
                if not math.isfinite(discrepancy) or discrepancy > numerics.QUERY_TOL:
                    raise ArithmeticError("posterior device query quadrature disagreement")
                probability = probabilities[0]
                diagnostics.update(
                    posterior_status=state.status,
                    max_logz_difference=state.log_z_discrepancy,
                    max_query_difference=discrepancy,
                )
        if not bool(torch.isfinite(probability).all()) or bool(((probability < 0) | (probability > 1)).any()):
            raise ArithmeticError("invalid adapted device probabilities")
        return {"probabilities": probability.cpu().numpy(), "diagnostics": diagnostics}


def numerical_case_status(
    dtype,
    probability_difference,
    *,
    objectives_pass,
    gradients_pass,
    convergence_pass,
    finite_pass,
    failure_dispositions_pass,
):
    """Apply the approved, separate CPU/CUDA precision contracts.

    Float64 preserves objective and gradient trajectories. Float32 is a
    separately qualified recipe: those trajectories remain visible diagnostics,
    while probability parity, finiteness and dispositions are mandatory gates.
    """
    if dtype not in {"float64", "float32"}:
        raise ValueError("unsupported numerical qualification precision")
    if (
        type(probability_difference) not in (int, float)
        or not math.isfinite(probability_difference)
        or probability_difference < 0
    ):
        raise ValueError("probability difference must be finite and nonnegative")
    checks = (objectives_pass, gradients_pass, convergence_pass, finite_pass, failure_dispositions_pass)
    if any(type(value) is not bool for value in checks):
        raise ValueError("explicit Boolean numerical qualification checks are required")
    trajectory_required = dtype == "float64"
    tolerance = 1e-8 if trajectory_required else 1e-5
    passed = (
        probability_difference <= tolerance
        and finite_pass
        and convergence_pass
        and failure_dispositions_pass
        and (not trajectory_required or (objectives_pass and gradients_pass))
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "qualification_contract": (
            "cpu_cuda_float64_preservation/v1" if trajectory_required else "cpu_cuda_float32_recipe/v1"
        ),
        "probability_atol": tolerance,
        "objectives_pass": objectives_pass,
        "gradients_pass": gradients_pass,
        "convergence_pass": convergence_pass,
        "finite_pass": finite_pass,
        "failure_dispositions_pass": failure_dispositions_pass,
        "trajectory_gate_applied": trajectory_required,
        "trajectory_diagnostic_only": not trajectory_required,
    }


def numerical_preflight():
    """One explicit real CPU/CUDA qualification, never invoked on import.

    Both precision classes execute all 300 optimizer updates for item,
    additive, bilinear and both support-residual heads. The approved float64
    preservation and float32 recipe contracts are separate. Float32 trajectory
    discrepancies remain diagnostics, including failed float64-style checks.
    This is a numerical backend qualification on synthetic data, not a
    predictive performance experiment.
    The caller must configure the pinned runtime and external resource guard.
    """
    torch, _ = _torch("cuda", "float64")
    rng = np.random.default_rng(20261030)
    features = rng.normal(0, 0.4, (32, 12))
    subjects = rng.normal(0, 0.4, (32, 4))
    labels = np.asarray([int((i * 7) % 5 < 2) for i in range(32)])
    pairs = ["large"] * 16 + ["small_a"] * 8 + ["small_b"] * 8
    episodes = []
    for i, n in enumerate((0, 3, 7)):
        episodes.append(
            {
                "base_logits": rng.normal(0, 0.4, 8),
                "y": labels[i * 8 : (i + 1) * 8],
                "support_features": features[:n],
                "support_labels": labels[:n],
                "pair_id": "episode_" + str(i),
                "weight": 1 / 3,
            }
        )
    result = {"schema": "proxy-gpu-numerical-preflight/v1", "steps": 300}
    for dtype, probability_atol in (("float64", 1e-8), ("float32", 1e-5)):
        cases = []
        for kind in ("item", "additive", "bilinear", "residual_counts", "residual_features"):
            heads, probabilities, timings = {}, {}, {}
            for device in ("cpu", "cuda"):
                torch.cuda.synchronize()
                started = time.monotonic()
                if kind.startswith("residual_"):
                    head = fit_residual(episodes, seed=17, kind=kind.split("_", 1)[1], device=device, dtype=dtype)
                    resident = ResidentResidual(head, device)
                    predictions = [
                        resident.predict(e["base_logits"], e["support_features"], e["support_labels"]) for e in episodes
                    ]
                    probability = np.concatenate(predictions)
                else:
                    subject = None if kind == "item" else subjects
                    head = fit_head(
                        features,
                        labels,
                        pairs,
                        seed=17,
                        kind=kind,
                        subject_features=subject,
                        device=device,
                        dtype=dtype,
                    )
                    probability = ResidentHead(head, device).predict(features, subject)
                torch.cuda.synchronize()
                timings[device] = time.monotonic() - started
                heads[device], probabilities[device] = head, probability
            objective = {
                device: np.asarray([r["objective"] for r in head["diagnostics"]["history"]])
                for device, head in heads.items()
            }
            gradient = {
                device: np.asarray([r["gradient_inf"] for r in head["diagnostics"]["history"]])
                for device, head in heads.items()
            }
            objectives_pass = bool(np.allclose(objective["cpu"], objective["cuda"], atol=1e-8, rtol=1e-6))
            gradients_pass = bool(np.allclose(gradient["cpu"], gradient["cuda"], atol=1e-8, rtol=1e-6))
            difference = float(np.max(np.abs(probabilities["cpu"] - probabilities["cuda"])))
            parameters_difference = max(
                float(
                    np.max(
                        np.abs(
                            np.asarray(heads["cpu"]["parameters"][name]) - np.asarray(heads["cuda"]["parameters"][name])
                        )
                    )
                )
                for name in heads["cpu"]["parameters"]
            )
            convergence_pass = heads["cpu"]["training"]["convergence"] == heads["cuda"]["training"]["convergence"]
            finite_pass = all(
                np.isfinite(value).all() for value in (*objective.values(), *gradient.values(), *probabilities.values())
            ) and all(
                np.isfinite(np.asarray(value)).all() for head in heads.values() for value in head["parameters"].values()
            )
            fallback_calls = {d: h["diagnostics"]["fallback_calls"] for d, h in heads.items()}
            failure_dispositions_pass = fallback_calls["cpu"] == fallback_calls["cuda"]
            contract = numerical_case_status(
                dtype,
                difference,
                objectives_pass=objectives_pass,
                gradients_pass=gradients_pass,
                convergence_pass=convergence_pass,
                finite_pass=finite_pass,
                failure_dispositions_pass=failure_dispositions_pass,
            )
            cases.append(
                {
                    **contract,
                    "kind": kind,
                    "steps": 300,
                    "max_probability_difference": difference,
                    "max_parameter_difference": parameters_difference,
                    "max_objective_difference": float(np.max(np.abs(objective["cpu"] - objective["cuda"]))),
                    "max_gradient_difference": float(np.max(np.abs(gradient["cpu"] - gradient["cuda"]))),
                    "convergence": {d: h["training"]["convergence"] for d, h in heads.items()},
                    "fallback_calls": fallback_calls,
                    "timings_seconds": timings,
                }
            )
        contract = numerical_case_status(
            dtype,
            max(c["max_probability_difference"] for c in cases),
            objectives_pass=all(c["objectives_pass"] for c in cases),
            gradients_pass=all(c["gradients_pass"] for c in cases),
            convergence_pass=all(c["convergence_pass"] for c in cases),
            finite_pass=all(c["finite_pass"] for c in cases),
            failure_dispositions_pass=all(c["failure_dispositions_pass"] for c in cases),
        )
        result[dtype] = {
            **contract,
            "max_probability_difference": max(c["max_probability_difference"] for c in cases),
            "probability_atol": probability_atol,
            "trajectory_atol": 1e-8,
            "trajectory_rtol": 1e-6,
            "cases": cases,
        }
    probes = []
    for update in ("none", "map", "posterior"):
        for support_y in ([], [1, 0, 1], [0, 0, 0], [1, 1, 1]):
            base = [-6.0, -0.2, 0.0, 0.0, 3.0, 6.0]
            support = [-0.4, 0.1, 0.5][: len(support_y)]
            cpu = adapt_probabilities(base, support, support_y, update=update, device="cpu")["probabilities"]
            gpu = adapt_probabilities(base, support, support_y, update=update, device="cuda")["probabilities"]
            difference = float(np.max(np.abs(cpu - gpu)))
            probes.append(
                {
                    "update": update,
                    "support_count": len(support_y),
                    "max_probability_difference": difference,
                    "status": "PASS" if difference <= 1e-8 else "FAIL",
                }
            )
    from research.proxy_policies.models import pooling_probabilities as reference_pooling

    query = np.vstack((np.zeros((1, 12)), features[:7]))
    support = np.vstack((np.zeros((1, 12)), features[8:14]))
    support_y = [0, 1, 0, 1, 1, 0, 1]
    reference = reference_pooling(query, support, support_y, tau=4.0)
    gpu = pooling_probabilities(query, support, support_y, device="cuda", dtype="float64")
    pooling_difference = float(np.max(np.abs(reference - gpu)))
    result["adaptation"] = {
        "cases": probes,
        "status": "PASS" if all(c["status"] == "PASS" for c in probes) else "FAIL",
    }
    result["pooling"] = {
        "max_probability_difference": pooling_difference,
        "status": "PASS" if pooling_difference <= 1e-8 else "FAIL",
    }
    result["status"] = (
        "PASS"
        if all(result[key]["status"] == "PASS" for key in ("float64", "float32", "adaptation", "pooling"))
        else "FAIL"
    )
    return result
