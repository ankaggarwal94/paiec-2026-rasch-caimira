"""Content-addressed cache of fitted predictors (one pickle per fold), so variants that change only
test-time behaviour skip the offline fit (~5 min per fold on the 208-benchmark universe).

Key = sha1 of
  - the predictor class name and every public, non-callable class/instance attribute (params),
    except those the class lists in RUNTIME_PARAMS (test-time only; re-applied after loading);
  - the source of every `fit` / `_fit*` method in the class MRO (plus the methods a class lists in FIT_HELPERS: fit-time
    helpers whose names do not start with `_fit`, e.g. IRT._fold_of), every module-level function in
    the modules defining those classes, and the whole of features/ and data/ (fit-time code);
  - the training benchmark list, the data config and the dataset revision.
Runtime methods (adapt, extend, score, _bench_step, ...) are *not* hashed: edits to them reuse the
cached fit, which is the point. If a fit depends on code outside these (rare), set
`cv.fit_cache: false` or bump the class attribute FIT_VERSION.
A subclass that changes only runtime behaviour can set `FIT_AS = <parent class>`: it then shares the
parent's cached fits (key and file as the parent; loaded fits are re-classed). It must not define fit /
_fit* methods or module-level functions, and all its new params must be RUNTIME_PARAMS.

Location: $PAIEC_FIT_CACHE; else the team's shared /fs/clip-scratch/mgor/paiec-cache/fits when it exists
(shared by worktrees; safe because the key covers the fit code); else data/fit_cache in the repo.
"""
import hashlib
import inspect
import json
import os
import pickle
import sys
from pathlib import Path

_PKG = Path(__file__).parent
_SHARED = Path("/fs/clip-scratch/mgor/paiec-cache/fits")
DEFAULT_DIR = _SHARED if _SHARED.parent.is_dir() else _PKG.parents[1] / "data" / "fit_cache"


def _params(pred):
    out = {}
    for cls in reversed(type(pred).__mro__):
        for k, v in vars(cls).items():
            if not k.startswith("_") and not callable(v) and not isinstance(v, (property, staticmethod, classmethod)):
                out[k] = v
    out.update({k: v for k, v in vars(pred).items() if not k.startswith("_")})
    runtime = set(getattr(pred, "RUNTIME_PARAMS", ()))
    base = getattr(type(pred), "FIT_AS", None)
    if base is not None:   # runtime-only subclass: key as the parent class
        out["name"] = base.name
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in sorted(out.items())
            if k not in runtime and k not in ("RUNTIME_PARAMS", "FIT_HELPERS")}


def _fit_source(pred):
    """Fit-time source behind pred's key. A FIT_AS class hashes its parent's (it shares the parent's fit by contract), so
    runtime-only helpers in the subclass's module (e.g. irt_item's _solve) do not split it from the parent's cache."""
    parts, mods = [], set()
    for cls in (getattr(type(pred), "FIT_AS", None) or type(pred)).__mro__:
        if cls is object:
            continue
        mods.add(cls.__module__)
        for k, v in vars(cls).items():
            if (k == "fit" or k.startswith("_fit") or k in vars(cls).get("FIT_HELPERS", ())) and callable(v):
                parts.append(inspect.getsource(v))
    for m in sorted(mods):
        mod = sys.modules[m]
        for _, fn in sorted(inspect.getmembers(mod, inspect.isfunction)):
            if fn.__module__ == m:
                parts.append(inspect.getsource(fn))
    for sub in ("features", "data"):
        for f in sorted((_PKG / sub).glob("*.py")):
            parts.append(f.read_text())
    return "\n".join(parts)


def cache_key(pred, train_benchmarks, data_cfg, revision) -> str:
    cls = getattr(type(pred), "FIT_AS", None) or type(pred)
    blob = json.dumps({"cls": cls.__name__, "params": _params(pred), "train": sorted(train_benchmarks),
                       "data": data_cfg, "rev": revision}, sort_keys=True, default=str)
    return hashlib.sha1((blob + _fit_source(pred)).encode()).hexdigest()[:20]


def fit_cached(pred, train, train_benchmarks, data_cfg, revision, enabled=True, verbose=True):
    """Return pred fitted on train, loading/saving a cached fit when enabled."""
    if not enabled:
        return pred.fit(train)
    key = cache_key(pred, train_benchmarks, data_cfg, revision)
    base = getattr(type(pred), "FIT_AS", None)
    path = Path(os.environ.get("PAIEC_FIT_CACHE", DEFAULT_DIR)) / f"{(base or type(pred)).__name__}_{key}.pkl"
    runtime = {k: getattr(pred, k) for k in getattr(pred, "RUNTIME_PARAMS", ())}
    if path.exists():
        try:
            with open(path, "rb") as f:
                fitted = pickle.load(f)
            if base is not None:   # a parent-class fit: adopt the runtime subclass
                fitted.__class__ = type(pred)
            for k, v in runtime.items():
                setattr(fitted, k, v)
            fitted.reset_runtime()
            if verbose:
                print(f"    fit cache hit {path.name}")
            return fitted
        except Exception as e:  # corrupt / incompatible pickle: refit
            print(f"    fit cache unreadable ({e!r}); refitting")
    pred.fit(train)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    cls = type(pred)
    if base is not None:   # store as the parent class, so parent-class runs load a plain parent
        pred.__class__ = base
    try:
        with open(tmp, "wb") as f:
            pickle.dump(pred, f, protocol=pickle.HIGHEST_PROTOCOL)
    finally:
        pred.__class__ = cls
    os.replace(tmp, path)
    return pred
