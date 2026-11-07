"""Private P2/P3 numerical draft: no datasets, models, imports or fitting.

Support construction is a numerical contract, not source/label admission.
The caller must bind legal same-pair realized occurrences, frozen heads/priors,
and full visible identities. All results and their nested containers are immutable.
No optional query, reference probe or call order can mutate production state.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from functools import lru_cache

VERSION = "p2-p3-numerics/v1"
DIMENSIONS = (1, 2)
ORDERS = (21, 41)
REFERENCE_ORDER = 81
MAX_SUPPORT = 31
MAP_STEPS = 50
MAP_GRAD_TOL = 1e-8
ARMIJO = 1e-4
MAX_HALVINGS = 20
LOGZ_TOL = 1e-6
QUERY_TOL = 1e-5
REFERENCE_QUERY_TOL = 1e-7


def _number(value, name):
    if type(value) not in (int, float):
        raise ValueError(name + " must be a finite Python int/float, excluding bool")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(name + " is outside finite float64 range") from exc
    if not math.isfinite(result):
        raise ValueError(name + " must be finite")
    return result


def _vector(value, length, name):
    if type(value) not in (list, tuple) or len(value) != length:
        raise ValueError(name + " has the wrong dimension")
    return tuple(_number(x, name) for x in value)


def _relevance(value, dimension):
    result = _vector(value, dimension, "relevance")
    if result[0] != 1.0 or (dimension == 2 and abs(result[1]) > 2.0):
        raise ValueError("relevance requires intercept 1 and second coordinate in [-2,2]")
    return result


def _sha(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def sigmoid(value):
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    e = math.exp(value)
    return e / (1.0 + e)


def softplus(value):
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


def _dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b, strict=True))


def _finite(values):
    return all(math.isfinite(x) for x in values)


@dataclass(frozen=True)
class Support:
    base_logits: tuple[float, ...]
    relevance: tuple[tuple[float, ...], ...]
    labels: tuple[int, ...]
    prior_mean: tuple[float, ...]
    sha256: str

    @property
    def dimension(self):
        return len(self.prior_mean)


def prepare_support(base_logits, relevance, labels, prior_mean):
    """Freeze exact numerical values; preserve every supplied support occurrence."""
    if type(prior_mean) not in (list, tuple) or len(prior_mean) not in DIMENSIONS:
        raise ValueError("prior mean must have dimension one or two")
    mu = _vector(prior_mean, len(prior_mean), "prior mean")
    if any(type(value) not in (list, tuple) for value in (base_logits, relevance, labels)):
        raise ValueError("support columns must be explicit lists/tuples")
    if not len(base_logits) == len(relevance) == len(labels) or len(labels) > MAX_SUPPORT:
        raise ValueError("support columns differ or exceed 31")
    if any(type(y) is not int or y not in (0, 1) for y in labels):
        raise ValueError("labels must be exact binary integers")
    b = tuple(_number(x, "base logit") for x in base_logits)
    r = tuple(_relevance(row, len(mu)) for row in relevance)
    y = tuple(labels)
    identity = _sha({"version": VERSION, "base_logits": b, "relevance": r, "labels": y, "prior_mean": mu})
    return Support(b, r, y, mu, identity)


def _validate_support(support):
    if type(support) is not Support:
        raise ValueError("Support from prepare_support required")
    reconstructed = prepare_support(support.base_logits, support.relevance, support.labels, support.prior_mean)
    if reconstructed != support:
        raise ValueError("support identity or immutable container changed")


def _energy(theta, support):
    terms = [0.5 * (x - m) ** 2 for x, m in zip(theta, support.prior_mean, strict=True)]
    for b, r, y in zip(support.base_logits, support.relevance, support.labels, strict=True):
        z = math.fsum((b, _dot(r, theta)))
        terms.append(softplus(-z if y else z))
    value = math.fsum(terms)
    if not math.isfinite(value):
        raise ArithmeticError("nonfinite objective")
    return value


def _derivatives(theta, support):
    d = support.dimension
    residuals, curvatures = [], []
    for b, r, y in zip(support.base_logits, support.relevance, support.labels, strict=True):
        z = math.fsum((b, _dot(r, theta)))
        residuals.append(-sigmoid(-z) if y else sigmoid(z))
        curvatures.append(sigmoid(z) * sigmoid(-z))
    gradient = tuple(
        math.fsum(
            [theta[j] - support.prior_mean[j]] + [e * r[j] for e, r in zip(residuals, support.relevance, strict=True)]
        )
        for j in range(d)
    )
    hessian = tuple(
        tuple(
            math.fsum([float(j == k)] + [w * r[j] * r[k] for w, r in zip(curvatures, support.relevance, strict=True)])
            for k in range(d)
        )
        for j in range(d)
    )
    objective = _energy(theta, support)
    if not _finite((*gradient, *(x for row in hessian for x in row))):
        raise ArithmeticError("nonfinite derivatives")
    return objective, gradient, hessian


def objective_gradient_hessian(theta, support):
    _validate_support(support)
    point = _vector(theta, support.dimension, "theta")
    return _derivatives(point, support)


def _eigenvalues(matrix):
    if len(matrix) == 1:
        return (matrix[0][0],)
    a, b, c = matrix[0][0], matrix[0][1], matrix[1][1]
    distance = math.hypot(a - c, 2 * b)
    return ((a + c - distance) / 2, (a + c + distance) / 2)


def _cholesky(matrix):
    if len(matrix) == 1:
        if matrix[0][0] <= 0:
            raise ArithmeticError("nonpositive curvature")
        return ((math.sqrt(matrix[0][0]),),)
    if matrix[0][1] != matrix[1][0] or matrix[0][0] <= 0:
        raise ArithmeticError("invalid symmetric curvature")
    l00 = math.sqrt(matrix[0][0])
    l10 = matrix[1][0] / l00
    residual = matrix[1][1] - l10 * l10
    if residual <= 0 or not math.isfinite(residual):
        raise ArithmeticError("nonpositive curvature")
    return ((l00, 0.0), (l10, math.sqrt(residual)))


def _solve_cholesky(lower, rhs):
    if len(rhs) == 1:
        return (rhs[0] / lower[0][0] ** 2,)
    u0 = rhs[0] / lower[0][0]
    u1 = (rhs[1] - lower[1][0] * u0) / lower[1][1]
    x1 = u1 / lower[1][1]
    x0 = (u0 - lower[1][0] * x1) / lower[0][0]
    return (x0, x1)


def _inverse_transpose(lower, t):
    """L^(-T)t, not L^(-1)t, for H=L L^T."""
    if len(t) == 1:
        return (t[0] / lower[0][0],)
    x1 = t[1] / lower[1][1]
    return ((t[0] - lower[1][0] * x1) / lower[0][0], x1)


@dataclass(frozen=True)
class SupportDiagnostics:
    count: int
    dimension: int
    gram_eigenvalues: tuple[float, ...]
    gram_rank: int
    rank_relative_tolerance: float
    relevance_variance: float | None
    hessian_eigenvalues: tuple[float, ...] | None
    hessian_condition: float | None
    data_curvature_eigenvalues: tuple[float, ...] | None
    prior_curvature_fractions: tuple[float, ...] | None
    prior_displacement: float | None


def _diagnostics(support, theta=None, hessian=None):
    d = support.dimension
    gram = tuple(tuple(math.fsum(r[j] * r[k] for r in support.relevance) for k in range(d)) for j in range(d))
    eigen = _eigenvalues(gram)
    tolerance = 1e-10 * max(1.0, max(eigen))
    rank = sum(x > tolerance for x in eigen)
    variance = None
    if d == 2 and support.labels:
        mean = math.fsum(r[1] for r in support.relevance) / len(support.labels)
        variance = math.fsum((r[1] - mean) ** 2 for r in support.relevance) / len(support.labels)
    h_eigen = None if hessian is None else _eigenvalues(hessian)
    return SupportDiagnostics(
        len(support.labels),
        d,
        eigen,
        rank,
        1e-10,
        variance,
        h_eigen,
        None if h_eigen is None else max(h_eigen) / min(h_eigen),
        None if h_eigen is None else tuple(x - 1 for x in h_eigen),
        None if h_eigen is None else tuple(1 / x for x in h_eigen),
        None
        if theta is None
        else math.sqrt(math.fsum((x - m) ** 2 for x, m in zip(theta, support.prior_mean, strict=True))),
    )


@dataclass(frozen=True)
class MapResult:
    support_sha256: str
    status: str
    reason: str | None
    mode: tuple[float, ...] | None
    objective: float | None
    gradient: tuple[float, ...] | None
    hessian: tuple[tuple[float, ...], ...] | None
    iterations: int
    gradient_inf: float | None
    diagnostics: SupportDiagnostics


def _map_failure(support, reason, iterations, point=None, values=None):
    if values is None:
        return MapResult(
            support.sha256, "FALLBACK", reason, None, None, None, None, iterations, None, _diagnostics(support)
        )
    objective, gradient, hessian = values
    return MapResult(
        support.sha256,
        "FALLBACK",
        reason,
        point,
        objective,
        gradient,
        hessian,
        iterations,
        max(abs(x) for x in gradient),
        _diagnostics(support, point, hessian),
    )


def solve_map(support):
    """At most 50 Newton updates and 20 halvings per update; no restart."""
    _validate_support(support)
    point = support.prior_mean
    last = None
    last_point = None
    try:
        for iteration in range(MAP_STEPS + 1):
            values = _derivatives(point, support)
            last = values
            last_point = point
            objective, gradient, hessian = values
            eigen = _eigenvalues(hessian)
            if min(eigen) < 1 - 1e-10 or max(eigen) > 1 + 5 * len(support.labels) / 4 + 1e-10:
                return _map_failure(support, "map_curvature_bound", iteration, point, values)
            grad_inf = max(abs(x) for x in gradient)
            if grad_inf <= MAP_GRAD_TOL:
                return MapResult(
                    support.sha256,
                    "OK",
                    None,
                    point,
                    objective,
                    gradient,
                    hessian,
                    iteration,
                    grad_inf,
                    _diagnostics(support, point, hessian),
                )
            if iteration == MAP_STEPS:
                return _map_failure(support, "map_nonconvergence", iteration, point, values)
            direction = tuple(-x for x in _solve_cholesky(_cholesky(hessian), gradient))
            slope = _dot(gradient, direction)
            if not math.isfinite(slope) or slope >= 0:
                return _map_failure(support, "map_non_descent", iteration, point, values)
            accepted = None
            for halvings in range(MAX_HALVINGS + 1):
                alpha = 2.0**-halvings
                proposed = tuple(math.fsum((x, alpha * step)) for x, step in zip(point, direction, strict=True))
                if _energy(proposed, support) <= objective + ARMIJO * alpha * slope:
                    accepted = proposed
                    break
            if accepted is None:
                return _map_failure(support, "map_line_search", iteration, point, values)
            point = accepted
    except (ArithmeticError, ValueError):
        return _map_failure(support, "map_nonfinite_or_curvature", iteration, last_point, last)
    raise AssertionError("bounded solver fell through")


def _hermite_value_previous(order, x):
    previous, current = 0.0, 1.0
    for n in range(1, order + 1):
        previous, current = current, math.sqrt(2.0 / n) * x * current - math.sqrt((n - 1.0) / n) * previous
    return current, previous


@lru_cache(maxsize=3)
def hermite_rule(order):
    """Physicists' GH roots via orthogonal-polynomial interlacing/bisection.

    Weights are sqrt(pi)/(n*p_(n-1)(root)^2), with p_0=1 and normalized
    recurrence. No NumPy/SciPy/Torch runtime dependency or tabulated asset.
    """
    if type(order) is not int or order not in (21, 41, 81):
        raise ValueError("only frozen GH orders 21, 41 and 81 are supported")
    roots = []
    for n in range(1, order + 1):
        bound = math.sqrt(2 * n + 1)
        boundaries = [-bound, *roots, bound]
        current = []
        for lo, hi in zip(boundaries[:-1], boundaries[1:], strict=True):
            left, right = _hermite_value_previous(n, lo)[0], _hermite_value_previous(n, hi)[0]
            if left == 0 or right == 0 or (left > 0) == (right > 0):
                raise ArithmeticError("Hermite interlacing bracket failed")
            for _ in range(80):
                midpoint = (lo + hi) / 2
                if midpoint in (lo, hi):
                    break
                value = _hermite_value_previous(n, midpoint)[0]
                if value == 0:
                    lo = hi = midpoint
                    break
                if (value > 0) == (left > 0):
                    lo, left = midpoint, value
                else:
                    hi = midpoint
            current.append((lo + hi) / 2)
        for j in range(n // 2):
            absolute = (current[-j - 1] - current[j]) / 2
            current[j], current[-j - 1] = -absolute, absolute
        if n % 2:
            current[n // 2] = 0.0
        roots = current
    weights = tuple(math.sqrt(math.pi) / (order * _hermite_value_previous(order, x)[1] ** 2) for x in roots)
    if not _finite((*roots, *weights)) or min(weights) <= 0 or abs(math.fsum(weights) - math.sqrt(math.pi)) > 2e-13:
        raise ArithmeticError("invalid Hermite weights")
    return tuple(roots), weights


@dataclass(frozen=True)
class QuadratureGrid:
    order: int
    points: tuple[tuple[float, ...], ...]
    normalized_weights: tuple[float, ...]
    log_z: float
    correction_range: tuple[float, float]
    effective_nodes: float


def _integrate_grid(mode, hessian, energy, order):
    lower = _cholesky(hessian)
    nodes, weights = hermite_rule(order)
    points, logs, corrections = [], [], []
    for indices in itertools.product(range(order), repeat=len(mode)):
        t = tuple(nodes[i] for i in indices)
        offset = _inverse_transpose(lower, t)
        point = tuple(math.fsum((m, math.sqrt(2) * x)) for m, x in zip(mode, offset, strict=True))
        correction = -energy(point) + math.fsum(x * x for x in t)
        points.append(point)
        corrections.append(correction)
        logs.append(math.fsum(math.log(weights[i]) for i in indices) + correction)
    if not _finite((*logs, *corrections, *(x for point in points for x in point))):
        raise ArithmeticError("nonfinite quadrature")
    maximum = max(logs)
    exponentials = tuple(math.exp(value - maximum) for value in logs)
    total = math.fsum(exponentials)
    normalized = tuple(x / total for x in exponentials)
    jacobian = len(mode) / 2 * math.log(2) - math.fsum(math.log(lower[j][j]) for j in range(len(mode)))
    log_z = maximum + math.log(total) + jacobian
    if not math.isfinite(log_z) or abs(math.fsum(normalized) - 1) > 1e-14:
        raise ArithmeticError("invalid quadrature normalization")
    return QuadratureGrid(
        order,
        tuple(points),
        normalized,
        log_z,
        (min(corrections), max(corrections)),
        1.0 / math.fsum(x * x for x in normalized),
    )


@dataclass(frozen=True)
class StateResult:
    support: Support
    map_result: MapResult
    status: str
    reason: str | None
    production: QuadratureGrid | None
    check: QuadratureGrid | None
    log_z_discrepancy: float | None


def prepare_posterior(support, map_result=None):
    _validate_support(support)
    mode = solve_map(support) if map_result is None else map_result
    if type(mode) is not MapResult or mode.support_sha256 != support.sha256:
        raise ValueError("MAP result belongs to a different support state")
    if mode.status != "OK":
        return StateResult(support, mode, "FALLBACK", mode.reason, None, None, None)
    try:
        verified = _derivatives(_vector(mode.mode, support.dimension, "MAP mode"), support)
        if verified != (mode.objective, mode.gradient, mode.hessian) or max(abs(x) for x in verified[1]) > MAP_GRAD_TOL:
            raise ValueError("supplied MAP values differ")
        production = _integrate_grid(mode.mode, mode.hessian, lambda t: _energy(t, support), ORDERS[0])
        check = _integrate_grid(mode.mode, mode.hessian, lambda t: _energy(t, support), ORDERS[1])
        discrepancy = abs(production.log_z - check.log_z)
        if discrepancy > LOGZ_TOL:
            return StateResult(
                support, mode, "FALLBACK", "state_quadrature_logz_disagreement", production, check, discrepancy
            )
        return StateResult(support, mode, "OK", None, production, check, discrepancy)
    except (ArithmeticError, OverflowError, ValueError):
        return StateResult(support, mode, "FALLBACK", "state_quadrature_nonfinite_or_invalid", None, None, None)


@dataclass(frozen=True)
class QueryResult:
    probability: float
    status: str
    reason: str | None
    probability_by_order: tuple[tuple[int, float], ...] = ()
    discrepancy: float | None = None


def _query(base_logit, relevance, dimension):
    return _number(base_logit, "query base logit"), _relevance(relevance, dimension)


def predict_map(base_logit, relevance, map_result):
    if type(map_result) is not MapResult:
        raise ValueError("MapResult required")
    if map_result.status != "OK":
        return QueryResult(0.5, "FALLBACK", map_result.reason)
    try:
        b, r = _query(base_logit, relevance, len(map_result.mode))
        z = math.fsum((b, _dot(r, map_result.mode)))
        if not math.isfinite(z):
            raise ArithmeticError("nonfinite query logit")
        return QueryResult(sigmoid(z), "OK", None)
    except (ArithmeticError, OverflowError, ValueError):
        return QueryResult(0.5, "FALLBACK", "query_nonfinite_or_invalid")


def _probability(b, r, grid):
    terms = []
    for point, weight in zip(grid.points, grid.normalized_weights, strict=True):
        z = math.fsum((b, _dot(r, point)))
        if not math.isfinite(z):
            raise ArithmeticError("nonfinite posterior query logit")
        terms.append(weight * sigmoid(z))
    value = math.fsum(terms)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ArithmeticError("invalid posterior probability")
    return value


def predict_posterior(base_logit, relevance, state):
    if type(state) is not StateResult:
        raise ValueError("StateResult required")
    if state.status != "OK":
        return QueryResult(0.5, "FALLBACK", state.reason)
    try:
        b, r = _query(base_logit, relevance, state.support.dimension)
        p21 = _probability(b, r, state.production)
        p41 = _probability(b, r, state.check)
        discrepancy = abs(p21 - p41)
        values = ((21, p21), (41, p41))
        if discrepancy > QUERY_TOL:
            return QueryResult(0.5, "FALLBACK", "query_quadrature_disagreement", values, discrepancy)
        return QueryResult(p21, "OK", None, values, discrepancy)
    except (ArithmeticError, OverflowError, ValueError):
        return QueryResult(0.5, "FALLBACK", "query_nonfinite_or_invalid")


@dataclass(frozen=True)
class ReferenceState:
    production_state: StateResult
    reference: QuadratureGrid | None
    status: str
    reason: str | None
    log_z_discrepancies: tuple[tuple[str, float], ...] = ()


def prepare_reference(state):
    """Explicit blinded-probe work only; never changes production state."""
    if type(state) is not StateResult:
        raise ValueError("StateResult required")
    if state.status != "OK":
        return ReferenceState(state, None, "FALLBACK", state.reason)
    try:
        grid = _integrate_grid(
            state.map_result.mode, state.map_result.hessian, lambda t: _energy(t, state.support), REFERENCE_ORDER
        )
        errors = (
            ("21_vs_81", abs(state.production.log_z - grid.log_z)),
            ("41_vs_81", abs(state.check.log_z - grid.log_z)),
        )
        if max(x for _, x in errors) > LOGZ_TOL:
            return ReferenceState(state, grid, "FALLBACK", "reference_logz_disagreement", errors)
        return ReferenceState(state, grid, "OK", None, errors)
    except (ArithmeticError, OverflowError, ValueError):
        return ReferenceState(state, None, "FALLBACK", "reference_nonfinite_or_invalid")


def reference_probe(base_logit, relevance, state, reference=None):
    """Return checks only. Reference values never replace production forecasts."""
    if reference is None:
        reference = prepare_reference(state)
    if type(reference) is not ReferenceState or reference.production_state != state:
        raise ValueError("reference belongs to a different production state")
    if reference.status != "OK":
        return QueryResult(0.5, "FALLBACK", reference.reason)
    try:
        b, r = _query(base_logit, relevance, state.support.dimension)
        p21 = _probability(b, r, state.production)
        p41 = _probability(b, r, state.check)
        p81 = _probability(b, r, reference.reference)
        values = ((21, p21), (41, p41), (81, p81))
        error = abs(p41 - p81)
        if error > REFERENCE_QUERY_TOL or abs(p21 - p81) > QUERY_TOL or abs(p21 - p41) > QUERY_TOL:
            return QueryResult(0.5, "FALLBACK", "reference_probability_disagreement", values, error)
        return QueryResult(p21, "OK", None, values, error)
    except (ArithmeticError, OverflowError, ValueError):
        return QueryResult(0.5, "FALLBACK", "query_nonfinite_or_invalid")
