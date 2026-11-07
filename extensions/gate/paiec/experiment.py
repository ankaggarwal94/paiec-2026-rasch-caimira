"""Config -> leave-one-benchmark-out replay -> runs/<name>/{config.yaml, metrics.json, pairs.csv}.

Headline metric `alc` = mean over test benchmarks (folds) of the pair-macro ALC, averaged over
seeds. Also reported: `alc_pairs` (pooled pair-macro over all folds, closer to the hosted
average when benchmarks have many pairs), per-budget Brier, ECE-ALC, per-fold detail, timing.
"""
import csv
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import yaml

from .acquisition import build_policy
from .fitcache import fit_cached
from .metrics import BUDGETS
from .predictors import build_predictor
from .sim.replay import build_pairs, replay


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False,
                              cwd=Path(__file__).parent).stdout.strip()
    except OSError:
        return ""


def eval_subjects(ds, cv) -> set | None:
    """Subject indices eligible for evaluation (None = all).
    cv.eval_subjects: a file of private subject ids, one per line (fixes the eval population across data
    variants; every run writes its own as runs/<name>/eval_subjects.txt). Else cv.eval_seen_from: subjects
    whose model identity appears in these benchmarks."""
    if cv.get("eval_subjects"):
        ids = set(Path(cv["eval_subjects"]).read_text().split())
        return {k for k, sid in enumerate(ds.subject_ids) if sid in ids}
    if cv.get("eval_seen_from"):
        from .features.subject import subject_identity
        ref = np.isin(ds.resp_b, [ds.bench_names.index(b) for b in cv["eval_seen_from"] if b in ds.bench_names])
        seen = {subject_identity(ds.subjects[s]) for s in np.unique(ds.resp_s[ref])}
        return {k for k, s in enumerate(ds.subjects) if subject_identity(s) in seen}
    return None


def run_experiment(cfg: dict, ds=None, write=True, verbose=True) -> dict:
    if ds is None:
        from .data.dataset import load_from_config
        ds = load_from_config(cfg["data"])
    benches = [b for b in (cfg["data"]["benchmarks"] or ds.benchmarks()) if b in ds.benchmarks()]
    folds = cfg["cv"]["test_benchmarks"] or benches
    cv, sim = cfg["cv"], cfg["sim"]
    allowed = eval_subjects(ds, cv)
    train_pool = cv.get("train_benchmarks")
    if isinstance(train_pool, str):   # a data directory: every benchmark downloaded there (e.g. data/aims/train)
        from .data.dataset import find_benchmarks
        train_pool = sorted(find_benchmarks(train_pool))
    thr = set(cfg["data"].get("thresholded_benchmarks") or [])   # thresholded-only benchmarks: trained on only if cv.train_thresholded
    from .features.subject import subject_identity
    ident = None
    hold_mask = None
    if cv.get("holdout_identities_file"):   # subject holdout: these model identities are removed from every training benchmark
        ident = np.array([subject_identity(sj) for sj in ds.subjects])
        hold_mask = np.isin(ident, json.loads(Path(cv["holdout_identities_file"]).read_text()))
    # the fit cache must see the subject holdout (it changes the training set without changing the benchmark list)
    fit_data = {**cfg["data"], "holdout_identities": int(hold_mask.sum()) if hold_mask is not None else 0,
                "holdout_ids_sha": __import__("hashlib").sha1(Path(cv["holdout_identities_file"]).read_bytes()).hexdigest()[:12]
                if cv.get("holdout_identities_file") else ""}
    rows, folds_out, timing = [], {}, {"fit_s": 0.0, "acq_calls": 0, "acq_s": 0.0, "eval_inputs": 0, "eval_s": 0.0}
    for fold in folds:
        group = [fold] if isinstance(fold, str) else list(fold)
        group = [b for b in group if b in benches]
        test_b = group[0] if len(group) == 1 else "+".join(group)
        train_b = [b for b in (train_pool or benches) if b in benches and b not in group
                   and (cv.get("train_thresholded") or b not in thr)]
        train = ds.restrict(train_b)
        if hold_mask is not None:
            train = train.drop_subjects(hold_mask)
        if ident is None:
            ident = np.array([subject_identity(sj) for sj in ds.subjects])
        seen_ident = {ident[k] for k in np.unique(train.resp_s)}
        sid_ident = dict(zip(ds.subject_ids, ident))
        t0 = time.perf_counter()
        pred = fit_cached(build_predictor(cfg["predictor"]), train, train_b, fit_data, ds.revision,
                          enabled=cv.get("fit_cache", True), verbose=verbose)
        timing["fit_s"] += time.perf_counter() - t0
        seed_alcs, seed_budget = [], []
        for seed in cv["seeds"]:
            pairs = build_pairs(ds, group, cv["split_seed"], seed, cv["min_items"],
                                cv["max_subjects_per_benchmark"], cv["max_eval_items_per_pair"], allowed)
            if not pairs:
                break
            pred.reset_runtime()
            res, t = replay(pred, build_policy(cfg["acquisition"]), ds, pairs, sim["evidence_scope"],
                            sim["fresh_each_checkpoint"], pair_order=sim.get("pair_order", "sorted"),
                            order_seed=(seed, test_b))
            for k in t:
                timing[k] += t[k]
            for r in res:
                r["seed"] = seed
                r["seen"] = int(sid_ident[r["subject_id"]] in seen_ident)   # model identity present in this fold's training set
            rows += res
            seed_alcs.append(np.mean([r["alc"] for r in res]))
            seed_budget.append([np.mean([r[f"B{n}"] for r in res]) for n in BUDGETS])
        if not seed_alcs:
            if verbose:
                print(f"  {test_b}: no eligible pairs, skipped")
            continue
        folds_out[test_b] = {"alc": float(np.mean(seed_alcs)), "alc_seed_std": float(np.std(seed_alcs)),
                             "n_pairs": len(pairs), "by_budget": dict(zip(map(str, BUDGETS), np.mean(seed_budget, 0).tolist()))}
        if verbose:
            print(f"  {test_b:>18}: ALC {folds_out[test_b]['alc']:.4f} ± {folds_out[test_b]['alc_seed_std']:.4f} "
                  f"({len(pairs)} pairs)")
    bench_alc = {}
    for r in rows:
        bench_alc.setdefault(r["benchmark"], []).append(r["alc"])
    metrics = {
        "name": cfg["name"],
        "alc": float(np.mean([f["alc"] for f in folds_out.values()])),
        "alc_pairs": float(np.mean([r["alc"] for r in rows])),
        "ece_alc": float(np.mean([r["ece_alc"] for r in rows])),
        "by_budget": {str(n): float(np.mean([f["by_budget"][str(n)] for f in folds_out.values()])) for n in BUDGETS},
        "folds": folds_out,
        "benchmarks": {b: {"alc": float(np.mean(v)), "n": len(v)} for b, v in sorted(bench_alc.items())},
        "strata": {name: {"alc": float(np.mean([r["alc"] for r in rows if r["seen"] == flag])),
                          "n": sum(r["seen"] == flag for r in rows)}
                   for name, flag in (("seen", 1), ("unseen", 0)) if any(r["seen"] == flag for r in rows)},
        "timing": timing,
        "predictor": cfg["predictor"], "acquisition": cfg["acquisition"], "sim": sim,
        "data_revision": ds.revision, "git_sha": _git_sha(),
    }
    if write:
        out = Path(cfg["out_root"]) / cfg["name"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        if allowed is not None:
            (out / "eval_subjects.txt").write_text("\n".join(sorted(ds.subject_ids[k] for k in allowed)) + "\n")
        with open(out / "pairs.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    if verbose:
        print(f"{cfg['name']}: ALC {metrics['alc']:.4f} (pairs {metrics['alc_pairs']:.4f}) | "
              + " ".join(f"B{n}={metrics['by_budget'][str(n)]:.3f}" for n in BUDGETS))
    return metrics
