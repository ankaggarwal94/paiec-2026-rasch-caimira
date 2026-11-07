"""Submission runtime: (de)serialise fitted predictors as JSON and wrap predict safely.

JSON instead of pickle so the ZIP does not depend on the hosted numpy version matching ours.
Only objects from the `paiec` package are reconstructed.
"""
import importlib
import json
import math
import os
import sys

import numpy as np

STATE_FILE = "predictor.json"
POLICY_FILE = "acquisition.json"


def _enc(x):
    if isinstance(x, np.ndarray):
        return {"__nd__": x.tolist(), "dtype": str(x.dtype)}
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, tuple):
        return {"__tuple__": [_enc(v) for v in x]}
    if isinstance(x, list):
        return [_enc(v) for v in x]
    if isinstance(x, dict):
        if all(isinstance(k, str) for k in x):
            return {k: _enc(v) for k, v in x.items()}
        return {"__items__": [[_enc(k), _enc(v)] for k, v in x.items()]}
    if type(x).__module__.startswith("paiec."):
        state = x.__getstate__() if hasattr(x, "__getstate__") else x.__dict__
        return {"__obj__": f"{type(x).__module__}:{type(x).__name__}", "state": _enc(state)}
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    raise TypeError(f"cannot serialise {type(x)}")


def _dec(x):
    if isinstance(x, list):
        return [_dec(v) for v in x]
    if not isinstance(x, dict):
        return x
    if "__nd__" in x:
        return np.array(x["__nd__"], dtype=x["dtype"])
    if "__tuple__" in x:
        return tuple(_dec(v) for v in x["__tuple__"])
    if "__items__" in x:
        return {_dec(k): _dec(v) for k, v in x["__items__"]}
    if "__obj__" in x:
        mod, cls = x["__obj__"].split(":")
        assert mod.startswith("paiec."), mod
        obj = object.__new__(getattr(importlib.import_module(mod), cls))
        obj.__dict__.update(_dec(x["state"]))
        return obj
    return {k: _dec(v) for k, v in x.items()}


def save_predictor(pred, path):
    with open(path, "w") as f:
        json.dump(_enc(pred), f)


def load_predictor(root):
    with open(os.path.join(root, STATE_FILE)) as f:
        return _dec(json.load(f))


def load_policy(root):
    from .acquisition import build_policy
    with open(os.path.join(root, POLICY_FILE)) as f:
        return build_policy(json.load(f))


_warned = False


def safe_predict(pred, input, labeled, fallback=0.5) -> float:
    """Never let one bad call fail the whole submission (failure codes 40/42)."""
    global _warned
    try:
        p = float(pred.predict(input, labeled))
        if math.isfinite(p) and 0.0 <= p <= 1.0:
            return p
        raise ValueError(f"bad probability {p}")
    except Exception as e:  # noqa: BLE001
        if not _warned:
            print(f"[paiec] predict failed, using fallback: {e!r}", file=sys.stderr)
            _warned = True
        return fallback
