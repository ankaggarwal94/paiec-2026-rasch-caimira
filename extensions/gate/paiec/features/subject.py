"""Subject featurizer and identity. Each entry in `fields` is one named extractor.

provider  hashed provider (+ model_config vendor prefix)
name      hashed tokens of normalized_name / source_model_name / model_config
think     reasoning-mode flag from names and reasoning_effort
size      log2 parameter count parsed from names ("70b", "1.5B"), plus missing flag
date      release_date as years since 2023, plus missing flag
effort    reasoning_effort ordinal, plus hashed value
harness   hashed harness and harness_version
"""
import re

import numpy as np

from .hashing import hash_tokens, parse_kv, tokenize
from .identity import model_identity

ALL_FIELDS = ("provider", "name", "think", "size", "date", "effort", "harness")
EFFORT = {"minimal": 0.1, "low": 0.25, "medium": 0.5, "high": 0.75, "xhigh": 1.0, "max": 1.0}
_THINK = re.compile(r"think|reason|\br1\b|\bo[134]\b|xhigh")
_SIZE = re.compile(r"(\d+(?:\.\d+)?)\s*b\b")
N_DENSE = 7


def _years_since_2023(d: str):
    try:
        years = int(d[:4]) - 2023
        return years + ((int(d[5:7]) - 1) / 12 if len(d) >= 7 else 0.0)
    except ValueError:
        return None


def names(subject: dict) -> list:
    extra = parse_kv(subject.get("subject_features_extra", ""))
    return [n for n in (subject.get("normalized_name", ""), extra.get("source_model_name", ""),
                        extra.get("model_config", "")) if n]


def subject_identity(subject: dict) -> str:
    """Canonical model identity shared across benchmarks (features/identity.py); falls back to the
    tokenised first visible name (or the whole dict) when no model name is visible."""
    canon = model_identity(subject)
    if canon:
        return canon
    ns = names(subject)
    base = ns[0] if ns else "|".join(subject.get(k, "") for k in sorted(subject))
    return " ".join(tokenize(base, 200))


def subject_key(subject: dict) -> tuple:
    """Exact visible identity (what the platform's pairing can be recovered from)."""
    return tuple(sorted(subject.items()))


class SubjectFeaturizer:
    def __init__(self, fields=ALL_FIELDS, dim=256):
        unknown = set(fields) - set(ALL_FIELDS)
        assert not unknown, f"unknown subject fields {unknown}"
        self.fields, self.dim = tuple(fields), dim

    @property
    def n_features(self) -> int:
        return self.dim + N_DENSE

    def transform_one(self, s: dict) -> np.ndarray:
        x = np.zeros(self.n_features, np.float32)
        h, dense = x[: self.dim], x[self.dim:]
        ns = names(s)
        text = " ".join(ns).lower()
        f = self.fields
        if "provider" in f:
            prov = s.get("provider", "") or (parse_kv(s.get("subject_features_extra", "")).get("model_config", "").split("/")[0])
            if prov:
                hash_tokens(h, [prov.lower()], "pv:")
        if "name" in f:
            hash_tokens(h, set(tokenize(text, 400)), "nm:", 0.5)
        if "harness" in f:
            for k in ("harness", "harness_version"):
                if s.get(k):
                    hash_tokens(h, [s[k].lower()], k[:3] + ":")
        effort = s.get("reasoning_effort", "").lower()
        if "think" in f:
            dense[0] = float(bool(_THINK.search(text)) or EFFORT.get(effort, 0) >= 0.5)
        if "size" in f:
            m = [float(v) for v in _SIZE.findall(text)]
            dense[1], dense[2] = (np.log2(max(m)) / 10, 0.0) if m else (0.0, 1.0)
        if "date" in f:
            years = _years_since_2023(s.get("release_date", ""))
            dense[3], dense[4] = (years, 0.0) if years is not None else (0.0, 1.0)
        if "effort" in f:
            dense[5], dense[6] = EFFORT.get(effort, 0.0), float(effort == "")
            if effort:
                hash_tokens(h, [effort], "ef:")
        return x

    def transform(self, subjects) -> np.ndarray:
        return np.stack([self.transform_one(s) for s in subjects]) if subjects else np.zeros((0, self.n_features), np.float32)
