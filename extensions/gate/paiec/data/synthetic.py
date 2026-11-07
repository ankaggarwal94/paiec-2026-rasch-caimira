"""Synthetic Rasch world in the Dataset format, for tests and quick debugging."""
import numpy as np

from .dataset import SUBJECT_FIELDS, Dataset, anon_benchmark_id

WORDS_EASY = ["add", "sum", "list", "print", "simple", "count"]
WORDS_HARD = ["prove", "optimize", "concurrent", "refactor", "theorem", "distributed"]


def make_synthetic(n_subjects=30, n_benchmarks=4, n_items=120, trials=1, coverage=1.0, seed=0) -> Dataset:
    rng = np.random.default_rng(seed)
    skill = rng.normal(0, 1, n_subjects)
    subjects = []
    for s in range(n_subjects):
        d = {f: "" for f in SUBJECT_FIELDS}
        d["normalized_name"] = f"Lab{s % 5} Model-{s} {int(10 * np.exp(skill[s] + 2))}b"
        d["provider"] = f"Lab{s % 5}"
        d["release_date"] = f"202{4 + s % 3}-0{1 + s % 9}-01"
        subjects.append(d)
    items, item_bench, item_ids, rs, ri, ry = [], [], [], [], [], []
    for b in range(n_benchmarks):
        offset = rng.normal(0, 1)
        for _ in range(n_items):
            diff = rng.normal(0, 1.2)
            words = rng.choice(WORDS_HARD if diff > 0 else WORDS_EASY, 4).tolist()
            idx = len(items)
            items.append({"item_content": " ".join(words) + f" task {idx}", "item_features": f"topic=t{idx % 3}",
                          "interactors": "", "benchmark_id": anon_benchmark_id(f"bench{b}")})
            item_bench.append(b)
            item_ids.append(f"i{idx}")
            for s in range(n_subjects):
                if rng.random() > coverage:
                    continue
                p = 1 / (1 + np.exp(-(skill[s] + offset - diff)))
                for _ in range(trials):
                    rs.append(s), ri.append(idx), ry.append(float(rng.random() < p))
    return Dataset(subjects, items, np.array(item_bench), [f"bench{b}" for b in range(n_benchmarks)],
                   [f"s{s}" for s in range(n_subjects)], item_ids, np.array(rs), np.array(ri),
                   np.array(ry), "synthetic")
