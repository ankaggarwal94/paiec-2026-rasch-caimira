"""Local replay of the hosted streaming protocol (approximate; see docs/evaluation.md).

Per (subject, benchmark) pair with >= min_items distinct items:
  - items are split 50/50 (floor to acquisition) with split_seed -> fixed pools across runs
  - acquisition candidates are shuffled with seed; each candidate's revealed outcome is one
    recorded trial drawn with seed (drawn up front, so it is matched across policies)
  - for each budget n: every pair streams candidates through predict + policy until it has n
    labels; then (optionally) the predictor's caches are reset, and every evaluation item is
    scored against *all* of its recorded trials.
evidence_scope="global" passes all revealed labels (every pair) to predict; "pair" passes only
the current pair's labels. Which one the platform does is unverified.
"""
import hashlib
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from ..metrics import ALC_WEIGHTS, BUDGETS


def _rng(*parts) -> random.Random:
    return random.Random(int(hashlib.md5(repr(parts).encode()).hexdigest()[:16], 16))


@dataclass
class Pair:
    subject: int
    bench: int
    acq: list
    outcomes: dict
    evl: list
    evl_ys: list
    labeled: list = field(default_factory=list)
    cursor: int = 0
    probs: dict = field(default_factory=dict)


def build_pairs(ds, benchmarks, split_seed=0, seed=0, min_items=80, max_subjects=None, max_eval_items=None,
                allowed_subjects=None):
    """Pairs with >= min_items distinct *source items* (variants of one item_id that differ only in
    interactors count once and stay in the same pool, as on the platform). max_subjects samples among
    eligible subjects only. allowed_subjects: optional set of subject indices eligible for evaluation.
    """
    bidx = {ds.bench_names.index(b) for b in benchmarks}
    rb = ds.resp_b
    cells = defaultdict(lambda: defaultdict(list))
    for s, i, y, b in zip(ds.resp_s, ds.resp_i, ds.resp_y, rb):
        if b in bidx and (allowed_subjects is None or s in allowed_subjects):
            cells[(int(s), int(b))][int(i)].append(float(y))
    groups = {}
    for key, items in cells.items():
        g = defaultdict(list)
        for i in sorted(items):
            g[ds.item_ids[i].split("|", 1)[0]].append(i)
        if len(g) >= min_items:
            groups[key] = sorted(g.values(), key=min)
    if max_subjects:
        keep = set()
        for b in bidx:
            subs = sorted({s for (s, bb) in groups if bb == b}, key=lambda s: ds.subject_ids[s])
            _rng("subjects", split_seed, b).shuffle(subs)
            keep |= {(s, b) for s in subs[:max_subjects]}
        groups = {k: v for k, v in groups.items() if k in keep}
    pairs = []
    for (s, b), gl in sorted(groups.items()):
        items = cells[(s, b)]
        gl = list(gl)
        _rng("split", split_seed, ds.subject_ids[s], b).shuffle(gl)
        h = len(gl) // 2
        acq_g, evl_g = gl[:h], gl[h:]
        evl = [i for v in evl_g for i in v]
        if max_eval_items and len(evl) > max_eval_items:
            evl = evl[:max_eval_items]
        r = _rng("order", seed, ds.subject_ids[s], b)
        r.shuffle(acq_g)
        acq, outcomes = [], {}
        for v in acq_g:   # one visible candidate per source item, then one recorded trial of it
            i = v[0] if len(v) == 1 else r.choice(v)
            acq.append(i)
            outcomes[i] = r.choice(items[i])
        pairs.append(Pair(s, b, acq, outcomes, evl, [np.array(items[i]) for i in evl]))
    return pairs


def _ece(probs, ys_list, bins=10):
    cnt, ps, ys = np.zeros(bins), np.zeros(bins), np.zeros(bins)
    for p, ys_ in zip(probs, ys_list):
        k = min(int(p * bins), bins - 1)
        cnt[k] += len(ys_); ps[k] += p * len(ys_); ys[k] += ys_.sum()
    return float(np.abs(ps - ys).sum() / cnt.sum())


def _order(pairs, how, n, seed):
    """Order in which pairs are advanced within a budget round (the platform's order is unknown).
    sorted: build order (subject index, benchmark); reverse; shuffle: seeded per round."""
    if how == "sorted":
        return pairs
    if how == "reverse":
        return pairs[::-1]
    if how == "shuffle":
        out = list(pairs)
        _rng("pair_order", seed, n).shuffle(out)
        return out
    raise ValueError(f"unknown pair_order {how!r}")


def replay(predictor, policy, ds, pairs, evidence_scope="global", fresh_each_checkpoint=True, per_call_eval=False,
           pair_order="sorted", order_seed=0):
    """per_call_eval: score with one predict() per input (hosted call pattern) instead of predict_batch.
    pair_order: order pairs are advanced within each budget round (sorted | reverse | shuffle)."""
    evidence = []
    if hasattr(policy, "bind"):
        policy.bind(predictor)
    t = {"acq_calls": 0, "acq_s": 0.0, "eval_inputs": 0, "eval_s": 0.0}
    for n in BUDGETS:
        for pr in _order(pairs, pair_order, n, order_seed):
            subj = ds.subjects[pr.subject]
            while len(pr.labeled) < n and pr.cursor < len(pr.acq):
                i = pr.acq[pr.cursor]
                inp = [subj, ds.items[i]]
                ev = evidence if evidence_scope == "global" else pr.labeled
                ctx = {"subject_id": ds.subject_ids[pr.subject], "benchmark_id": ds.items[i]["benchmark_id"],
                       "labels_acquired": len(pr.labeled), "labels_remaining": 31 - len(pr.labeled),
                       "max_labels": 31, "items_remaining": len(pr.acq) - pr.cursor}
                t0 = time.perf_counter()
                p = predictor.predict(inp, ev)
                take = policy(inp, p, ev, ctx)
                t["acq_s"] += time.perf_counter() - t0
                t["acq_calls"] += 1
                if take:
                    entry = [inp, int(pr.outcomes[i])]
                    pr.labeled.append(entry)
                    if evidence_scope == "global":
                        evidence.append(entry)
                pr.cursor += 1
        if fresh_each_checkpoint:
            predictor.reset_runtime()
        t0 = time.perf_counter()
        for pr in pairs:
            ev = evidence if evidence_scope == "global" else pr.labeled
            subj = ds.subjects[pr.subject]
            inputs = [(subj, ds.items[i]) for i in pr.evl]
            if per_call_eval:
                pr.probs[n] = np.array([predictor.predict(list(x), list(ev)) for x in inputs])
            else:
                pr.probs[n] = predictor.predict_batch(inputs, list(ev))
            t["eval_inputs"] += len(pr.evl)
        t["eval_s"] += time.perf_counter() - t0
    return score_pairs(ds, pairs), t


def score_pairs(ds, pairs):
    rows = []
    for pr in pairs:
        row = {"subject_id": ds.subject_ids[pr.subject], "benchmark": ds.bench_names[pr.bench],
               "n_acq": len(pr.acq), "n_eval_items": len(pr.evl),
               "n_eval_responses": int(sum(len(y) for y in pr.evl_ys)), "n_labels": len(pr.labeled),
               "acq_calls": pr.cursor}
        briers, eces = [], []
        for n in BUDGETS:
            p = pr.probs[n]
            se = sum(float(((pk - ys) ** 2).sum()) for pk, ys in zip(p, pr.evl_ys))
            briers.append(se / row["n_eval_responses"])
            eces.append(_ece(p, pr.evl_ys))
            row[f"B{n}"], row[f"ECE{n}"] = briers[-1], eces[-1]
        row["alc"] = float(np.dot(ALC_WEIGHTS, briers))
        row["ece_alc"] = float(np.dot(ALC_WEIGHTS, eces))
        rows.append(row)
    return rows
