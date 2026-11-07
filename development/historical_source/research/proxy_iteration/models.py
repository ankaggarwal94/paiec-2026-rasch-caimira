"""Dimension-independent, bounded direct heads and explicit-support prediction.

This successor does not mutate or reinterpret the frozen five-candidate fits.
The caller owns split admission, immutable feature identities and process limits.
"""

from __future__ import annotations

import math
import os
import time
from collections import Counter

import numpy as np

from research.roadmap import numerics

HEAD_SCHEMA = "proxy-iteration-head/v1"
MAX_DIMENSION = 16384
MAX_ROWS = 50000
MAX_FEATURE_BYTES = 1024**3


def _contains_boolean(values):
    if isinstance(values, (bool, np.bool_)):
        return True
    if isinstance(values, np.ndarray):
        return values.dtype.kind == "b"
    return isinstance(values, (list, tuple)) and any(_contains_boolean(v) for v in values)


def _matrix(values, dimension=None, *, name="features", allow_empty=True):
    if _contains_boolean(values):
        raise ValueError(name + " must not contain Boolean values")
    array = np.asarray(values)
    if array.dtype.kind not in "fiu" or array.dtype.kind == "b":
        raise ValueError(name + " must contain finite numeric values")
    if array.size == 0 and array.ndim == 1 and dimension is not None:
        array = np.empty((0, dimension), dtype=np.float64)
    if array.ndim != 2 or not 0 <= array.shape[1] <= MAX_DIMENSION:
        raise ValueError(name + " must be an N by D matrix")
    if dimension is not None and array.shape[1] != dimension:
        raise ValueError(name + " has a different feature dimension")
    if array.shape[0] > MAX_ROWS or (not allow_empty and array.shape[0] == 0):
        raise ValueError(name + " row count is outside the bounded contract")
    if array.size * 8 > MAX_FEATURE_BYTES:
        raise ValueError(name + " exceeds the one-GiB feature limit")
    result = np.asarray(array, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(name + " contains nonfinite values")
    return result


def _labels(values, count):
    if _contains_boolean(values):
        raise ValueError("labels must not contain Boolean values")
    array = np.asarray(values)
    if array.dtype.kind not in "fiu" or array.dtype.kind == "b":
        raise ValueError("labels must be numeric binary observations")
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.shape != (count,) or not np.isfinite(array).all() or not np.isin(array, (0, 1)).all():
        raise ValueError("one finite binary label is required per support/training row")
    return np.asarray(array, dtype=np.float64)


def validate_head(head):
    if not isinstance(head, dict) or head.get("schema") != HEAD_SCHEMA or head.get("kind") != "logistic":
        raise ValueError("unsupported direct-head schema")
    dimension = head.get("dimension")
    if type(dimension) is not int or not 0 <= dimension <= MAX_DIMENSION:
        raise ValueError("invalid head dimension")
    for key, length in (("beta", dimension + 1), ("center", dimension)):
        values = head.get(key)
        if (
            not isinstance(values, list)
            or len(values) != length
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
        ):
            raise ValueError("malformed head " + key)
    training = head.get("training")
    if not isinstance(training, dict) or training.get("convergence") not in {
        "FIXED",
        "CONVERGED",
        "BOUNDED_UNCONVERGED",
    }:
        raise ValueError("head requires an explicit finite-fit or fixed-control status")
    return dimension


def zero_head(dimension=0):
    """The zero-logit control, distinct from a fitted intercept-only head."""
    if type(dimension) is not int or not 0 <= dimension <= MAX_DIMENSION:
        raise ValueError("invalid fixed-head dimension")
    return {
        "schema": HEAD_SCHEMA,
        "kind": "logistic",
        "dimension": dimension,
        "beta": [0.0] * (dimension + 1),
        "center": [0.0] * dimension,
        "training": {"convergence": "FIXED", "recipe": "zero_logit_control"},
    }


def intercept_head(head):
    """Remove feature terms, preserving the fitted intercept for every row."""
    validate_head(head)
    return {
        "schema": HEAD_SCHEMA,
        "kind": "logistic",
        "dimension": 0,
        "beta": [float(head["beta"][0])],
        "center": [],
        "training": dict(head["training"], ablation="preserve_fitted_intercept_remove_features"),
    }


def fit_head(features, labels, pair_ids, *, seed=17, max_steps=300, ridge=0.01, max_seconds=1200):
    """Fit weighted logistic BCE with training-only centering and L2 penalty.

    Pair weights are equal; responses within a pair are equally weighted.
    The intercept receives the same ridge penalty as other coefficients,
    matching the prior direct-head recipe. The caller must impose a hard
    process lifetime: this wall check runs between optimizer updates.
    """
    x = _matrix(features, allow_empty=False)
    y = _labels(labels, len(x))
    if (
        not isinstance(pair_ids, (list, tuple))
        or len(pair_ids) != len(x)
        or any(not isinstance(p, str) or not p for p in pair_ids)
    ):
        raise ValueError("complete nonempty pair identities are required")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be a declared unsigned32-bit integer")
    if type(max_steps) is not int or not 1 <= max_steps <= 300:
        raise ValueError("fit requires1..300 explicit optimizer updates")
    if type(ridge) not in (int, float) or not math.isfinite(ridge) or not 0 <= ridge <= 100:
        raise ValueError("ridge must be finite and nonnegative")
    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 0 < max_seconds <= 1200:
        raise ValueError("fit wall bound must be positive and at most1200seconds")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    import torch

    if torch.cuda.is_initialized():
        raise ValueError("the bounded training worker must not initialize CUDA")
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    counts = Counter(pair_ids)
    weight = torch.tensor([1 / (len(counts) * counts[p]) for p in pair_ids], dtype=torch.float64)
    features_tensor = torch.tensor(x, dtype=torch.float64)
    center = (features_tensor * weight[:, None]).sum(dim=0)
    centered = features_tensor - center
    labels_tensor = torch.tensor(y, dtype=torch.float64)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    beta = torch.nn.Parameter(torch.randn(x.shape[1] + 1, generator=generator, dtype=torch.float64) * 0.01)
    optimizer = torch.optim.Adam(
        [beta],
        lr=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=False,
        foreach=False,
        fused=False,
    )
    history = []
    for step in range(max_steps + 1):
        if time.monotonic() - started > max_seconds:
            raise TimeoutError("bounded direct-head training deadline reached")
        optimizer.zero_grad(set_to_none=True)
        logits = beta[0] + centered @ beta[1:]
        bce = (
            weight * torch.nn.functional.binary_cross_entropy_with_logits(logits, labels_tensor, reduction="none")
        ).sum()
        penalty = 0.5 * ridge * beta.square().sum()
        objective = bce + penalty
        objective.backward()
        diagnostic = {
            "step": step,
            "objective": float(objective.detach()),
            "bce": float(bce.detach()),
            "ridge": float(penalty.detach()),
            "gradient_inf": float(beta.grad.abs().max()),
        }
        if not all(math.isfinite(v) for v in diagnostic.values()):
            raise ArithmeticError("nonfinite direct-head training state")
        history.append(diagnostic)
        if step < max_steps:
            optimizer.step()
    final = history[-1]
    previous = history[max(0, max_steps - 20)]
    relative = abs(final["objective"] - previous["objective"]) / max(1.0, abs(previous["objective"]))
    convergence = "CONVERGED" if final["gradient_inf"] <= 1e-4 and relative <= 1e-6 else "BOUNDED_UNCONVERGED"
    recipe = {
        "seed": seed,
        "steps": max_steps,
        "ridge": float(ridge),
        "optimizer": "Adam",
        "learning_rate": 0.01,
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "dtype": "float64",
        "device": "cpu",
        "threads": 1,
        "weighting": "equal_pair_then_equal_response_within_pair",
        "center": "training_only_pair_weighted",
        "intercept_regularized": True,
        "gradient_tolerance": 1e-4,
        "relative_objective_tolerance": 1e-6,
        "convergence": convergence,
    }
    head = {
        "schema": HEAD_SCHEMA,
        "kind": "logistic",
        "dimension": x.shape[1],
        "beta": beta.detach().tolist(),
        "center": center.tolist(),
        "training": recipe,
    }
    validate_head(head)
    return {
        "head": head,
        "diagnostics": {
            "history": history,
            "response_count": len(x),
            "pair_count": len(counts),
            "weight_sum": float(weight.sum()),
            "relative_objective_change": relative,
            "elapsed_seconds": time.monotonic() - started,
            "max_seconds": max_seconds,
            "deadline_semantics": "cooperative_between_updates_external_hard_limit_required",
            "fallback_calls": 0,
        },
    }


def _logits(head, values):
    return [
        math.fsum(
            [
                head["beta"][0],
                *(w * (float(x) - c) for w, x, c in zip(head["beta"][1:], row, head["center"], strict=True)),
            ]
        )
        for row in values
    ]


def predict(head, query_features, support_features, support_labels, *, update="none"):
    """Predict a query matrix after one explicit-support MAP/posterior solve.

    Inputs must already be restricted to the same complete pair by the replay
    caller. No hidden state, fitted-query centering or automatic fallback exists.
    """
    dimension = validate_head(head)
    query = _matrix(query_features, dimension, name="query features", allow_empty=False)
    support = _matrix(support_features, dimension, name="support features")
    labels = _labels(support_labels, len(support))
    if len(support) > 31 or update not in {"none", "map", "posterior"}:
        raise ValueError("declared update and at most31 same-pair support labels are required")
    bases = _logits(head, query)
    if not all(math.isfinite(x) for x in bases):
        raise ArithmeticError("nonfinite query logit")
    diagnostics = {
        "head_kind": "logistic",
        "head_convergence": head["training"]["convergence"],
        "update": update,
        "query_count": len(query),
        "support_count": len(support),
        "map_states": 0,
        "posterior_states": 0,
        "fallback_calls": 0,
        "max_logz_difference": 0.0,
        "max_query_difference": 0.0,
    }
    if update == "none":
        probabilities = [float(numerics.sigmoid(x)) for x in bases]
    else:
        state_input = numerics.prepare_support(
            _logits(head, support), [(1.0,)] * len(support), [int(y) for y in labels], (0.0,)
        )
        fit = numerics.solve_map(state_input)
        diagnostics.update(map_states=1, map_status=fit.status, map_reason=fit.reason, map_iterations=fit.iterations)
        if fit.status != "OK":
            raise ArithmeticError("MAP failure: " + fit.reason)
        state = fit
        if update == "posterior":
            state = numerics.prepare_posterior(state_input, fit)
            diagnostics.update(
                posterior_states=1, posterior_status=state.status, max_logz_difference=state.log_z_discrepancy
            )
            if state.status != "OK":
                raise ArithmeticError("posterior failure: " + state.reason)
        probabilities = []
        for base in bases:
            result = (
                numerics.predict_map(base, (1.0,), state)
                if update == "map"
                else numerics.predict_posterior(base, (1.0,), state)
            )
            if result.status != "OK":
                raise ArithmeticError("predictive numerical failure: " + result.reason)
            probabilities.append(float(result.probability))
            if update == "posterior":
                diagnostics["max_query_difference"] = max(diagnostics["max_query_difference"], result.discrepancy)
    if not all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities):
        raise ArithmeticError("invalid probability")
    return {"probabilities": probabilities, "diagnostics": diagnostics}
