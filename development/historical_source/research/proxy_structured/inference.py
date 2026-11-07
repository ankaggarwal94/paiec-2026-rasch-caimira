"""Gaussian-prior support inference for arbitrary scalar/rank-two 3PL loadings.

Only purchased support labels enter fit_state(s). Predictors accept no labels.
Caller provenance is mandatory outside this numeric interface; optional pair
identities add a local check. No training-subject lookup or silent fallback is
used. Finite bounded-search/quadrature checks are not global-identification
certificates.
"""

from __future__ import annotations

import math
import time
from functools import lru_cache

import numpy as np

SCHEMA = "proxy-structured-support/v1"
BOUND = 12.0
MAX_SUPPORT = 31


class InferenceFailure(RuntimeError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def _contains_boolean(value):
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray):
        return value.dtype.kind == "b"
    return isinstance(value, (list, tuple)) and any(_contains_boolean(part) for part in value)


def _array(value, ndim, name):
    array = np.asarray(value)
    if _contains_boolean(value) or array.ndim != ndim or array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError(name + " must be a finite numeric array of the declared rank")
    return array.astype(np.float64, copy=False)


def _parameters(offsets, loadings, floors, *, dimension=None):
    offsets = _array(offsets, 1, "offsets")
    raw_loadings = np.asarray(loadings)
    if raw_loadings.shape == (0,) and len(offsets) == 0 and isinstance(dimension, int) and 1 <= dimension <= 8:
        loadings = np.empty((0, dimension))
    loadings, floors = _array(loadings, 2, "loadings"), _array(floors, 1, "floors")
    if len(offsets) != len(loadings) or floors.shape != offsets.shape:
        raise ValueError("parameter row counts differ")
    if not 1 <= loadings.shape[1] <= 8 or (dimension is not None and loadings.shape[1] != dimension):
        raise ValueError("generic MAP supports rank1..8; this campaign qualifies rank1/2")
    if np.any((floors < 0) | (floors >= 1)):
        raise ValueError("floors must lie in [0,1)")
    return offsets, loadings, floors


def _request(request):
    required = {"offsets", "loadings", "floors", "labels", "prior_mean", "prior_cov"}
    if not required <= set(request) or set(request) - required - {"pair_id", "evidence_pair_ids", "method"}:
        raise ValueError("support request fields differ")
    mean = _array(request["prior_mean"], 1, "prior mean")
    offsets, loadings, floors = _parameters(
        request["offsets"], request["loadings"], request["floors"], dimension=len(mean)
    )
    labels = _array(request["labels"], 1, "labels")
    if labels.shape != offsets.shape or len(labels) > MAX_SUPPORT or not np.isin(labels, (0, 1)).all():
        raise ValueError("at most31 exact binary purchased labels required")
    mean, covariance = (
        _array(request["prior_mean"], 1, "prior mean"),
        _array(request["prior_cov"], 2, "prior covariance"),
    )
    rank = loadings.shape[1]
    if mean.shape != (rank,) or covariance.shape != (rank, rank):
        raise ValueError("prior and loading dimensions differ")
    if not np.allclose(covariance, covariance.T, atol=1e-12, rtol=0):
        raise ValueError("prior covariance must be symmetric")
    try:
        cholesky = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError("prior covariance must be positive definite") from error
    if np.linalg.cond(covariance) > 1e12:
        raise ValueError("prior covariance is numerically ill-conditioned")
    if "pair_id" in request or "evidence_pair_ids" in request:
        pair, pairs = request.get("pair_id"), request.get("evidence_pair_ids")
        if not isinstance(pair, str) or not pair or not isinstance(pairs, list) or len(pairs) != len(labels):
            raise ValueError("complete pair provenance required")
        if any(p != pair for p in pairs):
            raise ValueError("foreign-pair support label refused")
    return offsets, loadings, floors, labels, mean, covariance, cholesky


def _torch(device, dtype):
    if device not in ("cpu", "cuda") or dtype not in ("float32", "float64"):
        raise ValueError("device must be cpu/cuda and dtype float32/float64")
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no CPU fallback")
    return torch, getattr(torch, dtype)


def _evaluate(torch, points, offsets, loadings, floors, labels, mask):
    """Batched negative log posterior and exact analytic derivatives in white coordinates."""
    z = offsets[:, None, :] + torch.einsum("snk,smk->smn", loadings, points)
    log_floor = torch.where(floors > 0, torch.log(floors), -torch.inf)[:, None, :]
    positive = torch.logaddexp(log_floor, z) - torch.nn.functional.softplus(z)
    negative = torch.log1p(-floors)[:, None, :] - torch.nn.functional.softplus(z)
    log_likelihood = torch.where(labels[:, None, :] > 0, positive, negative)
    sigmoid = torch.sigmoid(z)
    upper_sigmoid = torch.sigmoid(z - log_floor)
    first = labels[:, None, :] * upper_sigmoid - sigmoid
    second = labels[:, None, :] * upper_sigmoid * (1 - upper_sigmoid) - sigmoid * (1 - sigmoid)
    active = mask[:, None, :]
    objective = 0.5 * points.square().sum(-1) - (log_likelihood * active).sum(-1)
    gradient = points - torch.einsum("smn,snk->smk", first * active, loadings)
    eye = torch.eye(points.shape[-1], dtype=points.dtype, device=points.device)
    hessian = eye - torch.einsum("smn,sni,snj->smij", second * active, loadings, loadings)
    return objective, gradient, hessian


def objective_gradient_hessian(
    theta, offsets, loadings, floors, labels, *, prior_mean, prior_cov, device="cpu", dtype="float64"
):
    """Public derivative probe in original ability coordinates, without fitting."""
    request = dict(
        offsets=offsets, loadings=loadings, floors=floors, labels=labels, prior_mean=prior_mean, prior_cov=prior_cov
    )
    o, loading, c, y, mean, covariance, chol = _request(request)
    values = _array(theta, 2, "ability probes")
    if values.shape[1] != len(mean):
        raise ValueError("ability probe dimension differs")
    torch, precision = _torch(device, dtype)

    def tensor(value):
        return torch.as_tensor(value, dtype=precision, device=device)

    white = np.linalg.solve(chol, (values - mean).T).T
    with torch.inference_mode():
        objective, gradient, hessian = _evaluate(
            torch,
            tensor(white)[None],
            tensor(o + loading @ mean)[None],
            tensor(loading @ chol)[None],
            tensor(c)[None],
            tensor(y)[None],
            torch.ones((1, len(y)), device=device, dtype=precision),
        )
    inverse = np.linalg.inv(chol)
    g = gradient[0].cpu().numpy() @ inverse
    h = np.einsum("ai,mab,bj->mij", inverse, hessian[0].cpu().numpy(), inverse)
    return {"objective": objective[0].cpu().numpy(), "gradient": g, "hessian": h}


def _cpu_white(request, white):
    """Independent NumPy/scipy scalar likelihood, not the Torch kernel."""
    from scipy.special import expit

    o, loading, floors, labels, mean, _, chol = _request(request)
    points = np.asarray(white, dtype=np.float64).reshape(-1, loading.shape[1])
    transformed = loading @ chol
    z = o[None] + (loading @ mean)[None] + points @ transformed.T
    with np.errstate(divide="ignore"):
        log_floor = np.log(floors)
        positive = np.logaddexp(log_floor[None], z) - np.logaddexp(0, z)
        negative = np.log1p(-floors)[None] - np.logaddexp(0, z)
    likelihood = np.where(labels[None] > 0, positive, negative)
    s, t = expit(z), expit(z - log_floor[None])
    value = 0.5 * np.square(points).sum(1) - likelihood.sum(1)
    gradient = points - (labels[None] * t - s) @ transformed
    return value, gradient


def scalar_map_reference(request, *, bound=BOUND, grid_size=4097):
    """Bounded dense derivative brackets plus all detected stationary roots.

    This comparison is independent of Torch/Newton. It does not certify the
    absence of arbitrarily narrow undetected modes for unrestricted loadings.
    """
    from scipy.optimize import brentq

    if _request(request)[1].shape[1] != 1:
        raise ValueError("scalar reference requires rank one")
    grid = np.linspace(-bound, bound, grid_size)
    _, derivative = _cpu_white(request, grid[:, None])
    derivative = derivative[:, 0]
    roots = []
    for i in range(len(grid) - 1):
        if derivative[i] == 0:
            roots.append(float(grid[i]))
        if derivative[i] * derivative[i + 1] < 0:
            roots.append(
                float(
                    brentq(lambda x: _cpu_white(request, [[x]])[1][0, 0], grid[i], grid[i + 1], xtol=1e-13, rtol=1e-14)
                )
            )
    candidates = sorted(set([-bound, bound, *roots]))
    values = _cpu_white(request, np.asarray(candidates)[:, None])[0]
    best = int(np.argmin(values))
    return {
        "white_map": candidates[best],
        "objective": float(values[best]),
        "stationary_points": roots,
        "bound": bound,
        "grid_size": grid_size,
        "boundary": abs(candidates[best]) >= bound - 1e-7,
        "scope": "bounded grid-bracketed scalar comparison; not an all-root certificate",
    }


def fit_states(
    requests,
    *,
    device="cuda",
    dtype="float64",
    method="map",
    max_iterations=100,
    bound=BOUND,
    max_seconds=120.0,
    verify_scalar="ambiguous",
    batch_size=128,
):
    """Batch independent MAP states on CUDA; preserve ambiguous/failed dispositions."""
    requests = list(requests)
    if method not in ("map", "posterior") or type(max_iterations) is not int or not 1 <= max_iterations <= 300:
        raise ValueError("unsupported method or iteration allowance")
    if not math.isfinite(bound) or not 4 <= bound <= 16 or not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("bounded finite inference limits required")
    if type(batch_size) is not int or not 1 <= batch_size <= 256:
        raise ValueError("batch size must be in1..256")
    if not requests:
        return []
    if verify_scalar not in (True, False, "3pl", "ambiguous"):
        raise ValueError("unsupported scalar comparison policy")
    normalized = [_request(request) for request in requests]
    torch, precision = _torch(device, dtype)
    started = time.monotonic()
    result = [None] * len(requests)
    tolerance = 1e-8 if dtype == "float64" else 1e-5
    for rank in sorted({row[1].shape[1] for row in normalized}):
        all_indices = [i for i, row in enumerate(normalized) if row[1].shape[1] == rank]
        for first in range(0, len(all_indices), batch_size):
            indices = all_indices[first : first + batch_size]
            size = len(indices)
            count = max(1, max(len(normalized[i][0]) for i in indices))
            arrays = [
                np.zeros((size, count)),
                np.zeros((size, count, rank)),
                np.zeros((size, count)),
                np.zeros((size, count)),
                np.zeros((size, count)),
            ]
            references = []
            scalar_seconds = time.monotonic()
            for si, index in enumerate(indices):
                o, loading, c, y, mean, _, chol = normalized[index]
                n = len(o)
                arrays[0][si, :n], arrays[1][si, :n] = o + loading @ mean, loading @ chol
                arrays[2][si, :n], arrays[3][si, :n], arrays[4][si, :n] = c, y, 1
                references.append(
                    scalar_map_reference(requests[index], bound=bound)
                    if rank == 1 and (verify_scalar is True or (verify_scalar == "3pl" and np.any(c > 0)))
                    else None
                )
            scalar_seconds = time.monotonic() - scalar_seconds
            tensors = [torch.as_tensor(a, dtype=precision, device=device) for a in arrays]
            if bool((tensors[2] >= 1).any()):
                raise ValueError("precision conversion rounded a floor to one")
            if rank == 1:
                starts = np.linspace(-bound, bound, 25)[:, None]
            elif rank == 2:
                axis = np.asarray([-8.0, -4.0, 0.0, 4.0, 8.0]).clip(-bound, bound)
                starts = np.asarray([(a, b) for a in axis for b in axis])
            else:
                starts = np.vstack(
                    [
                        np.zeros((1, rank)),
                        np.eye(rank) * 4,
                        -np.eye(rank) * 4,
                        np.full((1, rank), 4.0),
                        np.full((1, rank), -4.0),
                    ]
                )
            initial = np.broadcast_to(starts, (size, len(starts), rank)).copy()
            if rank == 1:
                for si, reference in enumerate(references):
                    if reference is not None:
                        initial[si, 0, 0] = reference["white_map"]
            points = torch.as_tensor(initial, dtype=precision, device=device)
            history = []
            with torch.inference_mode():
                for iteration in range(max_iterations):
                    if time.monotonic() - started > max_seconds:
                        raise TimeoutError("bounded support inference deadline reached")
                    objective, gradient, hessian = _evaluate(torch, points, *tensors)
                    if not bool(
                        torch.isfinite(objective).all()
                        and torch.isfinite(gradient).all()
                        and torch.isfinite(hessian).all()
                    ):
                        raise InferenceFailure("nonfinite support objective/derivatives", {"iteration": iteration})
                    history.append({"iteration": iteration, "best_objective": objective.min(1).values.cpu().tolist()})
                    eigen = torch.linalg.eigvalsh(hessian)[..., 0]
                    damping = torch.clamp(1e-5 - eigen, min=0)
                    eye = torch.eye(rank, device=device, dtype=precision)
                    direction = torch.linalg.solve(hessian + damping[..., None, None] * eye, gradient[..., None])[
                        ..., 0
                    ]
                    accepted = torch.zeros_like(objective, dtype=torch.bool)
                    updated = points.clone()
                    active = gradient.abs().amax(-1) > tolerance
                    for backtrack in range(20):
                        scale = 0.5**backtrack
                        proposed = (points - scale * direction).clamp(-bound, bound)
                        candidate, candidate_gradient, _ = _evaluate(torch, proposed, *tensors)
                        descend = (gradient * (points - proposed)).sum(-1)
                        slack = torch.finfo(precision).eps * 16 * (1 + objective.abs())
                        gradient_refinement = (
                            ((proposed - points).abs().amax(-1) <= 0.01)
                            & (candidate <= objective + slack)
                            & (candidate_gradient.abs().amax(-1) < gradient.abs().amax(-1) * 0.9)
                        )
                        take = (~accepted) & active & ((candidate <= objective - 1e-4 * descend) | gradient_refinement)
                        updated = torch.where(take[..., None], proposed, updated)
                        accepted |= take
                    movement = (updated - points).abs().amax()
                    points = updated
                    if not bool(active.any()) or float(movement.cpu()) < (1e-12 if dtype == "float64" else 1e-7):
                        break
                objective, gradient, hessian = _evaluate(torch, points, *tensors)
            values, positions = objective.cpu().numpy(), points.cpu().numpy()
            gradients, hessians = gradient.cpu().numpy(), hessian.cpu().numpy()
            for si, index in enumerate(indices):
                o, loading, c, y, mean, covariance, chol = normalized[index]
                order = np.argsort(values[si], kind="stable")
                minimum = float(values[si, order[0]])
                roundoff = np.finfo(np.float64 if dtype == "float64" else np.float32).eps * 32 * max(1.0, abs(minimum))
                eligible = [
                    int(j)
                    for j in order
                    if values[si, j] <= minimum + roundoff and np.linalg.eigvalsh(hessians[si, j])[0] > 0
                ]
                # No candidate at the minimum with positive curvature: the lowest point is reported, but a
                # stationary point without a positive-definite Hessian is never a converged MAP.
                best = (
                    min(eligible, key=lambda j: (float(np.max(np.abs(gradients[si, j]))), float(values[si, j])))
                    if eligible
                    else int(order[0])
                )
                positive_definite = bool(eligible)
                white_map, best_value = positions[si, best], float(values[si, best])
                gradient_norm = float(np.max(np.abs(gradients[si, best])))
                boundary = bool(np.max(np.abs(white_map)) >= bound - 1e-5)
                modes = []
                for j in order:
                    if np.max(np.abs(gradients[si, j])) > max(tolerance * 20, 1e-6):
                        continue
                    if np.linalg.eigvalsh(hessians[si, j])[0] <= 0:
                        continue
                    if not any(np.linalg.norm(positions[si, j] - mode["white"]) < 1e-3 for mode in modes):
                        modes.append({"white": positions[si, j].tolist(), "objective": float(values[si, j])})
                near = [mode for mode in modes if mode["objective"] - best_value <= 1e-7]
                status = (
                    "BOUNDARY_LIMIT"
                    if boundary
                    else "FAILED_CONVERGENCE"
                    if gradient_norm > max(tolerance * 20, 1e-6) or not positive_definite
                    else "AMBIGUOUS_MAP"
                    if len(near) > 1
                    else "CONVERGED"
                )
                reference = references[si]
                if (
                    rank == 1
                    and verify_scalar == "ambiguous"
                    and np.any(c > 0)
                    and (len(modes) > 1 or status != "CONVERGED")
                ):
                    reference = scalar_map_reference(requests[index], bound=bound)
                if reference is not None and (
                    reference["boundary"]
                    or best_value - reference["objective"] > (1e-8 if dtype == "float64" else 1e-4)
                ):
                    status = "FAILED_GLOBAL_COMPARISON"
                result[index] = {
                    "schema": SCHEMA,
                    "method": requests[index].get("method", method),
                    "rank": rank,
                    "dtype": dtype,
                    "support_count": len(y),
                    "evidence_pair_id": requests[index].get("pair_id"),
                    "pair_provenance_checked": "pair_id" in requests[index],
                    "offsets": o.tolist(),
                    "loadings": loading.tolist(),
                    "floors": c.tolist(),
                    "labels": y.astype(int).tolist(),
                    "prior_mean": mean.tolist(),
                    "prior_cov": covariance.tolist(),
                    "map": (mean + chol @ white_map).tolist(),
                    "white_map": white_map.tolist(),
                    "bound": bound,
                    "diagnostics": {
                        "status": status,
                        "objective": best_value,
                        "gradient_inf": gradient_norm,
                        "positive_definite": positive_definite,
                        "boundary": boundary,
                        "local_modes": modes,
                        "near_optimal_modes": len(near),
                        "multimodal": len(modes) > 1,
                        "iterations": len(history),
                        "scalar_reference": reference,
                        "scalar_reference_batch_seconds": scalar_seconds,
                        "device": device,
                        "no_fallback": True,
                        "empty_support_prior_only": len(y) == 0,
                        "scope": "bounded deterministic multistart; no general identification/global guarantee",
                    },
                }
                if result[index]["method"] not in ("map", "posterior"):
                    raise ValueError("unsupported per-state prediction method")
    return result


def fit_state(offsets, loadings, floors, labels, *, prior_mean, prior_cov, **kwargs):
    pair_id = kwargs.pop("pair_id", None)
    evidence_pair_ids = kwargs.pop("evidence_pair_ids", None)
    request = dict(
        offsets=offsets, loadings=loadings, floors=floors, labels=labels, prior_mean=prior_mean, prior_cov=prior_cov
    )
    if pair_id is not None or evidence_pair_ids is not None:
        request.update(pair_id=pair_id, evidence_pair_ids=evidence_pair_ids)
    return fit_states([request], **kwargs)[0]


@lru_cache(maxsize=16)
def _quadrature(rank, order, bound):
    from scipy.special import roots_legendre

    if rank not in (1, 2):
        raise ValueError("posterior integration is qualified only for rank1/2")
    x, w = roots_legendre(order)
    x, w = x * bound, w * bound
    if rank == 1:
        return x[:, None], np.log(w)
    nodes = np.asarray([(a, b) for a in x for b in x])
    return nodes, (np.log(w)[:, None] + np.log(w)[None]).ravel()


def _posterior(torch, precision, states, targets, order, device):
    rank = states[0]["rank"]
    bound = states[0]["bound"]
    nodes, log_weight = _quadrature(rank, order, bound)
    count = max(1, max(state["support_count"] for state in states))
    arrays = [
        np.zeros((len(states), count)),
        np.zeros((len(states), count, rank)),
        np.zeros((len(states), count)),
        np.zeros((len(states), count)),
        np.zeros((len(states), count)),
    ]
    for i, state in enumerate(states):
        o, loading, c, y, mean, _, chol = _request(
            {k: state[k] for k in ("offsets", "loadings", "floors", "labels", "prior_mean", "prior_cov")}
        )
        n = len(o)
        arrays[0][i, :n], arrays[1][i, :n] = o + loading @ mean, loading @ chol
        arrays[2][i, :n], arrays[3][i, :n], arrays[4][i, :n] = c, y, 1
    packed = [torch.as_tensor(value, device=device, dtype=precision) for value in arrays]
    white = torch.as_tensor(nodes, device=device, dtype=precision)
    log_w = torch.as_tensor(log_weight, device=device, dtype=precision)
    log_mass = []
    for begin in range(0, len(nodes), 2048):
        points = white[None, begin : begin + 2048].expand(len(states), -1, -1)
        log_mass.append(-_evaluate(torch, points, *packed)[0] + log_w[None, begin : begin + 2048])
    log_mass = torch.cat(log_mass, dim=1)
    probability_mass = torch.softmax(log_mass, dim=1)
    edge = torch.as_tensor(np.any(np.abs(nodes) > bound - 1, axis=1), device=device)
    boundary_mass = probability_mass[:, edge].sum(1).cpu().numpy()
    outputs = []
    for i, (state, target) in enumerate(zip(states, targets, strict=True)):
        o, loading, c = target
        mean, covariance = np.asarray(state["prior_mean"]), np.asarray(state["prior_cov"])
        chol = np.linalg.cholesky(covariance)
        transformed_o, transformed_l = o + loading @ mean, loading @ chol
        vector = []
        for start in range(0, len(o), 64):
            offset = torch.as_tensor(transformed_o[start : start + 64], device=device, dtype=precision)
            load = torch.as_tensor(transformed_l[start : start + 64], device=device, dtype=precision)
            floor = torch.as_tensor(c[start : start + 64], device=device, dtype=precision)
            p = floor[None] + (1 - floor[None]) * torch.sigmoid(offset[None] + white @ load.T)
            vector.append((probability_mass[i, :, None] * p).sum(0).cpu().numpy())
        outputs.append(np.concatenate(vector) if vector else np.empty(0))
    return outputs, boundary_mass


def predict_states(states, targets, *, device="cuda", dtype=None, return_diagnostics=False):
    """Batch MAP queries; posterior integrates true likelihood and checks refinement."""
    states, targets = list(states), list(targets)
    if len(states) != len(targets):
        raise ValueError("state/target counts differ")
    if not states:
        return []
    checked = []
    for state, target in zip(states, targets, strict=True):
        if state.get("schema") != SCHEMA or state["diagnostics"]["status"] not in ("CONVERGED", "AMBIGUOUS_MAP"):
            raise InferenceFailure("support state is not qualified", state.get("diagnostics", {}))
        if state["method"] == "map" and state["diagnostics"]["status"] == "AMBIGUOUS_MAP":
            raise InferenceFailure("ambiguous MAP state requires explicit posterior integration", state["diagnostics"])
        if set(target) != {"offsets", "loadings", "floors"}:
            raise ValueError("target parameters only; target labels are forbidden")
        checked.append(_parameters(target["offsets"], target["loadings"], target["floors"], dimension=state["rank"]))
    outputs, details = [None] * len(states), [None] * len(states)
    for rank in sorted({state["rank"] for state in states}):
        for method in ("map", "posterior"):
            for precision_name in ("float32", "float64"):
                indices = [
                    i
                    for i, state in enumerate(states)
                    if state["rank"] == rank
                    and state["method"] == method
                    and (dtype or state["dtype"]) == precision_name
                ]
                if not indices:
                    continue
                torch, precision = _torch(device, precision_name)
                with torch.inference_mode():
                    if method == "map":
                        maximum = max(1, max(len(checked[i][0]) for i in indices))
                        o, c = np.zeros((len(indices), maximum)), np.zeros((len(indices), maximum))
                        loading = np.zeros((len(indices), maximum, rank))
                        for si, i in enumerate(indices):
                            n = len(checked[i][0])
                            o[si, :n], loading[si, :n], c[si, :n] = checked[i]
                        theta = torch.as_tensor([states[i]["map"] for i in indices], device=device, dtype=precision)
                        z = torch.as_tensor(o, device=device, dtype=precision) + torch.einsum(
                            "sqk,sk->sq", torch.as_tensor(loading, device=device, dtype=precision), theta
                        )
                        floor = torch.as_tensor(c, device=device, dtype=precision)
                        values = (floor + (1 - floor) * torch.sigmoid(z)).cpu().numpy()
                        for si, i in enumerate(indices):
                            outputs[i] = values[si, : len(checked[i][0])]
                            details[i] = {"method": "map", "status": states[i]["diagnostics"]["status"]}
                    else:
                        orders = (128, 256, 512, 1024) if rank == 1 else (32, 64, 128, 256)
                        tolerance = 1e-8 if precision_name == "float64" else 1e-5
                        for begin in range(0, len(indices), 4):
                            subset = indices[begin : begin + 4]
                            previous = None
                            for order in orders:
                                values, boundary = _posterior(
                                    torch,
                                    precision,
                                    [states[i] for i in subset],
                                    [checked[i] for i in subset],
                                    order,
                                    device,
                                )
                                differences = (
                                    [math.inf] * len(subset)
                                    if previous is None
                                    else [
                                        float(np.max(np.abs(a - b))) if len(a) else 0.0
                                        for a, b in zip(values, previous, strict=True)
                                    ]
                                )
                                if previous is not None and max(differences) <= tolerance and max(boundary) <= 1e-8:
                                    for si, i in enumerate(subset):
                                        outputs[i] = values[si]
                                        details[i] = {
                                            "method": "posterior",
                                            "status": "CONVERGED_QUADRATURE",
                                            "order": order,
                                            "probability_refinement_difference": differences[si],
                                            "boundary_mass": float(boundary[si]),
                                        }
                                    break
                                previous = values
                            else:
                                raise InferenceFailure(
                                    "posterior integration did not converge",
                                    {
                                        "orders": list(orders),
                                        "differences": differences,
                                        "boundary_mass": boundary.tolist(),
                                    },
                                )
    if any(value is None or not np.isfinite(value).all() or np.any((value < 0) | (value > 1)) for value in outputs):
        raise InferenceFailure("invalid or unavailable prediction result", {"no_fallback": True})
    if return_diagnostics:
        return [
            {"probabilities": value.tolist(), "diagnostics": detail}
            for value, detail in zip(outputs, details, strict=True)
        ]
    return outputs


def predict_state(state, target_offsets, target_loadings, target_floors, **kwargs):
    return predict_states(
        [state], [dict(offsets=target_offsets, loadings=target_loadings, floors=target_floors)], **kwargs
    )[0]


def posterior_reference(request, target, *, bound=BOUND):
    """Independent adaptive scalar/rank-two scipy integration in prior-white space."""
    from scipy.integrate import dblquad, quad
    from scipy.optimize import minimize
    from scipy.special import expit

    o, loading, floors, _, mean, _, chol = _request(request)
    target_o, target_l, target_c = _parameters(
        target["offsets"], target["loadings"], target["floors"], dimension=loading.shape[1]
    )
    rank = loading.shape[1]
    if rank not in (1, 2):
        raise ValueError("independent adaptive integration supports rank1/2")
    if rank == 1:
        optimum = scalar_map_reference(request, bound=bound)
        shift = optimum["objective"]
    else:
        candidates = [
            minimize(
                lambda u: _cpu_white(request, [u])[0][0],
                np.asarray(start),
                jac=lambda u: _cpu_white(request, [u])[1][0],
                method="L-BFGS-B",
                bounds=[(-bound, bound)] * rank,
                options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 1000},
            )
            for start in ((0, 0), (-4, -4), (-4, 4), (4, -4), (4, 4))
        ]
        shift = min(float(candidate.fun) for candidate in candidates)

    def density(u):
        return math.exp(float(-_cpu_white(request, [u])[0][0] + shift))

    def integrate(function):
        if rank == 1:
            return quad(lambda u: function([u]), -bound, bound, epsabs=1e-10, epsrel=1e-10, limit=300)[0]
        return dblquad(
            lambda v, u: function([u, v]), -bound, bound, lambda _: -bound, lambda _: bound, epsabs=1e-9, epsrel=1e-9
        )[0]

    denominator = integrate(density)
    if not math.isfinite(denominator) or denominator <= 0:
        raise InferenceFailure("independent posterior normalization failed", {})
    predictions = []
    for offset, loading_row, floor in zip(target_o, target_l, target_c, strict=True):

        def integrand(u, offset=offset, loading_row=loading_row, floor=floor):
            theta = mean + chol @ np.asarray(u)
            return density(u) * (floor + (1 - floor) * expit(offset + loading_row @ theta))

        predictions.append(integrate(integrand) / denominator)
    return {
        "probabilities": predictions,
        "rank": rank,
        "bound": bound,
        "method": "independent adaptive scipy quad/dblquad",
        "normalization": denominator,
    }
