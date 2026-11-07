"""measurement-db tables -> Dataset in the competition's visible-input format.

Subjects and items are stored as the exact dicts `predict` receives (all strings,
missing -> ""), with an anonymised `benchmark_id`. True benchmark names and private
ids live only in parallel metadata lists, which predictors never see at score time.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

SUBJECT_FIELDS = ("normalized_name", "provider", "release_date", "access_date", "harness",
                  "harness_version", "reasoning_effort", "subject_features_extra")
ITEM_FIELDS = ("item_content", "item_features", "interactors", "benchmark_id")
TABLES = ("benchmarks", "subjects", "items", "responses")


def anon_benchmark_id(name: str) -> str:
    return "bm_" + hashlib.sha1(name.encode()).hexdigest()[:10]


@dataclass
class Dataset:
    subjects: list            # visible subject dicts; index = subject idx
    items: list               # visible item dicts; index = item idx
    item_bench: np.ndarray    # item idx -> benchmark idx
    bench_names: list         # private benchmark names
    subject_ids: list         # private subject ids (pair identity on the platform)
    item_ids: list            # private item ids
    resp_s: np.ndarray        # one entry per response (trial)
    resp_i: np.ndarray
    resp_y: np.ndarray        # binary, float64
    revision: str = ""

    @property
    def resp_b(self) -> np.ndarray:
        return self.item_bench[self.resp_i]

    def benchmarks(self) -> list:
        return [self.bench_names[b] for b in np.unique(self.resp_b)]

    def restrict(self, benchmarks) -> "Dataset":
        """Keep responses from these benchmarks; subject/item index spaces are unchanged."""
        keep = np.isin(self.resp_b, [self.bench_names.index(b) for b in benchmarks])
        return replace(self, resp_s=self.resp_s[keep], resp_i=self.resp_i[keep],
                       resp_y=self.resp_y[keep])

    def drop_subjects(self, subject_mask) -> "Dataset":
        """Remove all responses of the subjects flagged in `subject_mask` (bool array over `subjects`)."""
        keep = ~np.asarray(subject_mask, bool)[self.resp_s]
        return replace(self, resp_s=self.resp_s[keep], resp_i=self.resp_i[keep], resp_y=self.resp_y[keep])

    def summary(self) -> str:
        rows = []
        for b in np.unique(self.resp_b):
            m = self.resp_b == b
            rows.append(f"{self.bench_names[b]}: {m.sum()} responses, "
                        f"{len(np.unique(self.resp_s[m]))} subjects, "
                        f"{len(np.unique(self.resp_i[m]))} items, mean {self.resp_y[m].mean():.3f}")
        return "\n".join(rows)


def _s(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return str(v)


def parse_scale(raw) -> dict:
    try:
        return json.loads(raw) if isinstance(raw, str) else dict(raw or {})
    except (TypeError, ValueError):
        return {}


def _binarize(name, values, cfg, scale=None):
    """Map recorded grades to {0,1}. Returns (y, keep mask, rule used).

    cfg:
      mode          exclude (binary benchmarks only) | threshold
      threshold     cut on the scale normalised to [0, 1] by its declared bounds/values
                    (observed min/max for `mixed` or missing bounds); default 0.5
      per_benchmark {name: threshold | "exclude"} overrides
      orient        flip lower_is_better scales (incl. binary) so 1 = good outcome; default False
                    (keep recorded values, which is presumably what the platform scores)
    Unordered (nominal) and unbounded scales are excluded under threshold mode.
    """
    scale = scale or {}
    vals = np.asarray(values, float)
    finite = np.isfinite(vals)
    flip = bool(cfg.get("orient")) and scale.get("direction") == "lower_is_better"
    override = (cfg.get("per_benchmark") or {}).get(name)
    if override == "exclude":
        return vals, np.zeros(len(vals), bool), "excluded"
    if set(np.unique(vals[finite])) <= {0.0, 1.0}:
        return (1 - vals if flip else vals), finite, "binary"
    rule = override if override is not None else cfg.get("mode", "exclude")
    none = (vals, np.zeros(len(vals), bool), "excluded")
    if rule == "exclude" or scale.get("direction") == "unordered" or not finite.any():
        return none
    t = float(rule) if not isinstance(rule, str) else float(cfg.get("threshold", 0.5))
    lo, hi = scale.get("min"), scale.get("max")
    if scale.get("values"):
        lo, hi = min(scale["values"]), max(scale["values"])
    if scale.get("kind") == "interval" and (lo is None or hi is None):
        return none   # unbounded: no meaningful normalisation
    if lo is None or hi is None:
        lo, hi = float(vals[finite].min()), float(vals[finite].max())
    if hi <= lo:
        return none
    z = (vals - lo) / (hi - lo)
    if scale.get("direction") == "lower_is_better":
        z = 1 - z   # graded "lower is better" scales are oriented before thresholding
    y = (z >= t).astype(float)
    return y, finite, f"threshold@{t}"


def find_benchmarks(root) -> dict:
    """{benchmark slug: its formatted_tables dir} over one root or a list of roots (e.g. the train and
    val downloads). A slug present under several roots resolves to the first."""
    roots = [root] if isinstance(root, (str, Path)) else list(root)
    found = {}
    for r in map(Path, roots):
        for p in sorted(r.iterdir()):
            if (p / "formatted_tables").is_dir():
                found.setdefault(p.name, p / "formatted_tables")
    return found


def data_fingerprint(root, slugs) -> dict:
    """What the pickle cache must not confuse: where the tables came from, how big they are, and the loader code.
    Same revision + same config can still mean different data (a capped export vs the full db, a re-exported split)."""
    roots = [root] if isinstance(root, (str, Path)) else list(root)
    tables = find_benchmarks(roots)
    sizes = {s: [(tables[s] / f"{t}.parquet").stat().st_size for t in TABLES if (tables[s] / f"{t}.parquet").exists()]
             for s in slugs if s in tables}
    code = hashlib.sha1(b"".join((Path(__file__).parent.parent / p).read_bytes()
                                 for p in ("data/dataset.py", "features/subject.py", "features/identity.py"))).hexdigest()
    return {"roots": [str(Path(r).resolve()) for r in roots], "sizes": sizes, "code": code[:12]}


def read_revision(root) -> str:
    """Upstream measurement-db revision (the REVISION file); roots that disagree are joined with '+'."""
    roots = [root] if isinstance(root, (str, Path)) else list(root)
    revs = [(Path(r) / "REVISION").read_text().strip() for r in roots if (Path(r) / "REVISION").exists()]
    return "+".join(dict.fromkeys(revs))


def seen_identities(tables: dict, prefer_seen_from) -> set:
    """Canonical model identities of every subject in the `prefer_seen_from` benchmarks."""
    import pandas as pd
    from ..features.subject import subject_identity
    seen = set()
    for ref in prefer_seen_from or []:
        if ref in tables and (tables[ref] / "subjects.parquet").exists():
            for r in pd.read_parquet(tables[ref] / "subjects.parquet").itertuples(index=False):
                seen.add(subject_identity({k: _s(getattr(r, k, None)) for k in SUBJECT_FIELDS}))
    return seen


def cap_subjects(slug, subject_col, sub_tab, seen_ids, max_responses, n_items=None, min_items=0) -> np.ndarray:
    """Mask over responses keeping a seeded random subset of subjects with <= max_responses responses
    in total (each kept subject keeps all of its responses); subjects in `seen_ids` go first, then (when
    min_items) subjects with >= min_items distinct items (`n_items`: subject_id -> count)."""
    from ..features.subject import subject_identity
    if not max_responses or len(subject_col) <= max_responses:
        return np.ones(len(subject_col), bool)
    counts = subject_col.value_counts()
    seen = {r.subject_id for r in sub_tab.itertuples(index=False)
            if subject_identity({k: _s(getattr(r, k, None)) for k in SUBJECT_FIELDS}) in seen_ids}
    order = sorted(counts.index, key=lambda x: (x not in seen, bool(min_items) and n_items[x] < min_items,
                                                 hashlib.md5(f"{slug}|{x}".encode()).hexdigest()))
    keep_s, total = set(), 0
    for sid in order:
        if total and total + counts[sid] > max_responses:
            continue
        keep_s.add(sid)
        total += counts[sid]
    return subject_col.isin(keep_s).values


def load_from_config(d: dict, **kw) -> Dataset:
    """load_dataset from a config's `data` section."""
    return load_dataset(d["root"], d.get("benchmarks"), d.get("binarize"), d.get("cache_dir", "data/processed"),
                        max_responses=d.get("max_responses"), max_item_chars=d.get("max_item_chars"),
                        prefer_seen_from=d.get("prefer_seen_from"), cap_overrides=d.get("cap_overrides"),
                        exclude=d.get("exclude"), prefer_min_items=d.get("prefer_min_items", 0), **kw)


def load_dataset(root="data/raw", benchmarks=None, binarize=None, cache_dir="data/processed",
                 use_cache=True, max_responses=None, max_item_chars=None, prefer_seen_from=None,
                 cap_overrides=None, exclude=None, prefer_min_items=0) -> Dataset:
    """root: a measurement-db-layout directory ({benchmark}/formatted_tables/*.parquet) or a list of
    them. max_responses: cap per benchmark by keeping a seeded random subset of *subjects* (each kept
    pair keeps all of its items). max_item_chars: truncate stored item text (memory; featurizers
    read at most a few thousand characters anyway). prefer_seen_from: benchmarks whose model identities
    are kept first when capping (e.g. the six reference benchmarks), so capped benchmarks retain them.
    cap_overrides: {benchmark: cap or None (uncapped)} replacing max_responses for those benchmarks.
    exclude: benchmarks to skip even if present under root (e.g. a stale download of a since-removed benchmark).
    prefer_min_items: when capping, subjects with at least this many distinct items are kept before the others (so capped
    benchmarks keep eligible pairs). binarize.labels_dir: directory of per-benchmark label arrays (scripts/make_labels.py)
    row-aligned with responses.parquet; a benchmark with a labels file uses them (NaN = dropped row) instead of the
    threshold rule; binarize.labels_version is part of the cache key."""
    import pandas as pd

    binarize = binarize or {"mode": "exclude"}
    revision = read_revision(root)
    tables = find_benchmarks(root)
    slugs = sorted(tables)
    if benchmarks:
        slugs = [s for s in slugs if s in benchmarks]
    if exclude:
        slugs = [s for s in slugs if s not in set(exclude)]
    key_parts = ["v4", revision, slugs, binarize, max_responses, max_item_chars, prefer_seen_from,
                 data_fingerprint(root, slugs)]
    if cap_overrides:   # appended only when set, so existing caches keep their keys
        key_parts.append(cap_overrides)
    if exclude:
        key_parts.append({"exclude": sorted(exclude)})
    if prefer_min_items:
        key_parts.append({"prefer_min_items": prefer_min_items})
    key = hashlib.sha1(json.dumps(key_parts, sort_keys=True).encode()).hexdigest()[:12]
    cache = Path(cache_dir) / f"dataset_{key}.pkl"
    if use_cache and cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)

    subjects, subject_ids, sidx = [], [], {}
    items, item_ids, item_bench, iidx = [], [], [], {}
    rs, ri, ry, bench_names = [], [], [], []
    seen_ids = seen_identities(tables, prefer_seen_from)
    rules = {}
    for slug in slugs:
        t = tables[slug]
        if not all((t / f"{x}.parquet").exists() for x in TABLES):
            continue   # partially downloaded
        cat = pd.read_parquet(t / "benchmarks.parquet").iloc[0]
        if cat.get("release", "public") != "public" and not binarize.get("include_withheld"):
            continue
        resp = pd.read_parquet(t / "responses.parquet", columns=["subject_id", "item_id", "interactors", "response"])
        lab = Path(binarize["labels_dir"]) / f"{slug}.npy" if binarize.get("labels_dir") else None
        if lab is not None and lab.exists():
            y = np.load(lab).astype(float)
            assert len(y) == len(resp), f"{slug}: labels ({len(y)}) not aligned with responses ({len(resp)})"
            keep, rules[slug] = np.isfinite(y), "labels"
        else:
            y, keep, rules[slug] = _binarize(slug, resp.response.values, binarize, parse_scale(cat.get("response_scale")))
        if not keep.any():
            continue
        b = len(bench_names)
        bench_names.append(slug)
        anon = anon_benchmark_id(slug)
        for row in pd.read_parquet(t / "subjects.parquet").itertuples(index=False):
            if row.subject_id not in sidx:
                sidx[row.subject_id] = len(subjects)
                subjects.append({f: _s(getattr(row, f, None)) for f in SUBJECT_FIELDS})
                subject_ids.append(row.subject_id)
        resp = resp[keep]
        y = y[keep]
        cap = (cap_overrides or {}).get(slug, max_responses)
        if cap and len(resp) > cap:
            n_items = resp.groupby("subject_id").item_id.nunique() if prefer_min_items else None
            m = cap_subjects(slug, resp.subject_id, pd.read_parquet(t / "subjects.parquet"), seen_ids, cap,
                             n_items, prefer_min_items)
            resp, y = resp[m], y[m]
        # a visible item is (item, interactors): interactors live on responses and are part of the input
        inter = resp.interactors.fillna("").astype(str).values
        needed = set(resp.item_id.unique())
        meta = {}
        for row in pd.read_parquet(t / "items.parquet", columns=["item_id", "content", "item_features"]).itertuples(index=False):
            if row.item_id in needed and row.item_id not in meta:
                c = _s(row.content)
                meta[row.item_id] = (c[:max_item_chars] if max_item_chars else c, _s(row.item_features))
        idx = np.empty(len(resp), np.int64)
        for k, (iid, it) in enumerate(zip(resp.item_id.values, inter)):
            key = (iid, it)
            if key not in iidx:
                iidx[key] = len(items)
                content, feats = meta.get(iid, ("", ""))
                items.append({"item_content": content, "item_features": feats, "interactors": it, "benchmark_id": anon})
                item_ids.append(iid if not it else f"{iid}|{hashlib.sha1(it.encode()).hexdigest()[:8]}")
                item_bench.append(b)
            idx[k] = iidx[key]
        s_idx = resp.subject_id.map(sidx).values
        ok = ~np.isnan(s_idx.astype(float))   # drop responses whose subject is missing from subjects.parquet
        rs.append(s_idx[ok].astype(np.int64))
        ri.append(idx[ok])
        ry.append(y[ok])
    ds = Dataset(subjects, items, np.array(item_bench), bench_names, subject_ids, item_ids,
                 np.concatenate(rs).astype(np.int64), np.concatenate(ri).astype(np.int64),
                 np.concatenate(ry).astype(float), revision)
    ds.binarize_rules = rules
    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "wb") as f:
            pickle.dump(ds, f)
    return ds
