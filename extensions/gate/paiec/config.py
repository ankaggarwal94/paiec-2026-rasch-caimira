"""Experiment config: YAML file + dotted overrides (e.g. --set predictor.alpha=8)."""
import copy

import yaml

DEFAULTS = {
    "name": "unnamed",
    "out_root": "runs",
    "data": {"root": "data/raw", "benchmarks": None, "binarize": {"mode": "exclude"},
             "cache_dir": "data/processed",
             "max_responses": None, "max_item_chars": None, "prefer_seen_from": None, "cap_overrides": None,
             "exclude": None},
    "cv": {"fit_cache": True, "test_benchmarks": None, "train_benchmarks": None, "eval_seen_from": None, "eval_subjects": None, "seeds": [0, 1, 2], "split_seed": 0, "min_items": 80,
           "max_subjects_per_benchmark": None, "max_eval_items_per_pair": None},
    "sim": {"evidence_scope": "global", "fresh_each_checkpoint": True, "pair_order": "sorted"},
    "predictor": {"name": "empirical_mean"},
    "acquisition": {"name": "official_random"},
}


def merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def set_dotted(cfg: dict, assignment: str) -> None:
    key, _, raw = assignment.partition("=")
    *path, last = key.strip().split(".")
    node = cfg
    for p in path:
        node = node.setdefault(p, {})
    node[last] = yaml.safe_load(raw)


def _read(path) -> dict:
    """Read a YAML config, resolving `base: other.yaml` (paths relative to the repo/cwd)."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    return merge(_read(base), cfg) if base else cfg


def load_config(path=None, overrides=()) -> dict:
    cfg = merge(DEFAULTS, _read(path)) if path else copy.deepcopy(DEFAULTS)
    for o in overrides:
        set_dotted(cfg, o)
    return cfg
