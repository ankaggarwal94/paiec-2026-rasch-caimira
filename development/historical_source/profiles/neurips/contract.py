"""Validate the NeurIPS streaming contract without importing class-era code."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

__all__ = [
    "canonical_json",
    "load_config",
    "matching_labels",
    "normalize_evidence",
    "normalize_input",
    "pair_key",
    "subject_key",
    "validate_config",
    "validate_probability",
    "validate_published_input",
]

_DEFAULTS = {
    "predictor": "empirical_mean",
    "prior": 0.5,
    "alpha": 4.0,
    "acquisition": "uniform",
    "uncertainty_radius": 0.15,
}


def _json_value(value: Any) -> None:
    """Reject lossy JSON encodings, non-finite numbers, and Python-only values."""
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("JSON metadata must contain only finite numbers")
        return
    if isinstance(value, list):
        for element in value:
            _json_value(element)
        return
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        for element in value.values():
            _json_value(element)
        return
    raise ValueError("Metadata must contain JSON values only")


def canonical_json(value: Any) -> str:
    """Return lossless deterministic JSON, including every metadata field."""
    _json_value(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def subject_key(subject: dict) -> str:
    """Identity includes the complete visible subject, not only its model name."""
    if not isinstance(subject, dict):
        raise ValueError("Subject must be a dictionary")
    return canonical_json(subject)


def normalize_input(input: Any, require_benchmark: bool = False) -> tuple[dict, dict]:
    """Validate [subject, item], allowing the official zero-label smoke input.

    Runtime validates JSON metadata and preserves unknown fields rather than
    requiring unused predictor features. ``validate_published_input`` applies
    the complete published field schema to training and release fixtures.
    Missing anonymous benchmark IDs are permitted only when no evidence is
    pooled. This does not certify positive-budget deployed evaluator support.
    """
    if not isinstance(input, (list, tuple)) or len(input) != 2:
        raise ValueError("Input must be [subject, item]")
    subject, item = input
    subject_key(subject)
    if not isinstance(item, dict):
        raise ValueError("Item must be a dictionary")
    canonical_json(item)
    if require_benchmark:
        benchmark = item.get("benchmark_id")
        if not isinstance(benchmark, str) or not benchmark.strip():
            raise ValueError("An anonymous, nonempty item benchmark_id is required with evidence")
    return subject, item


def validate_published_input(input: Any, require_benchmark: bool = True) -> tuple[dict, dict]:
    """Require the current published string fields for training-data validation.

    Unknown JSON fields remain present so identity never silently discards
    additional subject metadata. Known fields may be empty strings, as in the
    official smoke fixture, but may not be missing or coerced to strings.
    """
    subject, item = normalize_input(input, require_benchmark=require_benchmark)
    subject_fields = (
        "normalized_name",
        "provider",
        "release_date",
        "access_date",
        "harness",
        "harness_version",
        "reasoning_effort",
        "subject_features_extra",
    )
    item_fields = ("item_content", "item_features", "interactors")
    for group, fields in ((subject, subject_fields), (item, item_fields)):
        for field in fields:
            if field not in group or not isinstance(group[field], str):
                raise ValueError(f"Published input field {field} must be present and a string")
    return subject, item


def pair_key(input: Any) -> tuple[str, str]:
    """Identify one complete subject–anonymous benchmark pair."""
    subject, item = normalize_input(input, require_benchmark=True)
    return subject_key(subject), item["benchmark_id"]


def validate_probability(value: Any, name: str = "prediction") -> float:
    """Return a built-in finite float in [0,1]; reject coercion of bad values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite probability")
    try:
        probability = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite probability") from exc
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError(f"{name} must be a finite probability in [0, 1]")
    return probability


def normalize_evidence(labeled: Any) -> list[tuple[list[dict], float]]:
    """Validate all global evidence, preserving independent repeated responses.

    Never collapse identical visible descriptions: they can represent separate
    response trials. Malformed evidence fails even when it belongs to a
    different subject, avoiding silently incomplete adaptation.
    """
    if labeled is None:
        return []
    if not isinstance(labeled, (list, tuple)):
        raise ValueError("Labeled evidence must be a sequence of [[subject, item], binary] entries")
    result = []
    for observation in labeled:
        if not isinstance(observation, (list, tuple)) or len(observation) != 2:
            raise ValueError("Each labeled entry must be [[subject, item], binary]")
        observed_input, label = observation
        subject, item = normalize_input(observed_input, require_benchmark=True)
        if isinstance(label, bool) or not isinstance(label, (int, float)) or label not in (0, 1):
            raise ValueError("Acquired responses must be binary numbers (0 or 1)")
        result.append(([subject, item], float(label)))
    return result


def matching_labels(input: Any, labeled: Any) -> list[float]:
    """Select evidence matching both full subject identity and benchmark ID."""
    evidence = normalize_evidence(labeled)
    normalize_input(input, require_benchmark=bool(evidence))
    if not evidence:
        return []
    target = pair_key(input)
    return [label for observed_input, label in evidence if pair_key(observed_input) == target]


def validate_config(config: Any = None) -> dict:
    """Normalize a small explicit configuration shared by both entrypoints."""
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise ValueError("NeurIPS configuration must be an object")
    if set(config) - set(_DEFAULTS):
        raise ValueError("Unknown NeurIPS configuration option")
    result = {**_DEFAULTS, **config}
    if result["predictor"] not in ("constant", "empirical_mean", "smoothed"):
        raise ValueError("Unknown NeurIPS predictor")
    result["prior"] = validate_probability(result["prior"], "prior")
    alpha = result["alpha"]
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError("alpha must be a finite nonnegative number")
    try:
        alpha = float(alpha)
    except (ValueError, OverflowError) as exc:
        raise ValueError("alpha must be a finite nonnegative number") from exc
    if not math.isfinite(alpha) or alpha < 0 or (result["predictor"] == "smoothed" and alpha == 0):
        raise ValueError("alpha must be finite and strictly positive for smoothed prediction")
    result["alpha"] = alpha
    if result["acquisition"] not in ("uniform", "uncertainty"):
        raise ValueError("Unknown NeurIPS acquisition policy")
    radius = validate_probability(result["uncertainty_radius"], "uncertainty_radius")
    if radius > 0.5:
        raise ValueError("uncertainty_radius must be in [0, 0.5]")
    result["uncertainty_radius"] = radius
    return result


def load_config(directory: Path) -> dict:
    """Load optional packaged config; invalid existing files always fail closed."""
    path = directory / "neurips_config.json"
    if not path.exists():
        return validate_config()
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("Cannot read NeurIPS configuration") from exc
    return validate_config(config)
