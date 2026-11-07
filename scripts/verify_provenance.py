"""Verify the selected-state receipt chain and the documented source overlap."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = ROOT / "data/provenance"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if condition is False:
        raise ValueError(message)


def main() -> None:
    receipt = read_json(PROVENANCE / "submission.receipt.json")
    refit = read_json(PROVENANCE / "refit.json")
    freeze = read_json(PROVENANCE / "freeze.json")
    combined = read_json(PROVENANCE / "combined-sources.json")
    catalog = read_json(PROVENANCE / "catalog-freeze-v6.json")
    panel = read_json(ROOT / "data/local/panel_manifest.json")
    state = ROOT / "runtime/candidate/caimira-requires-state.json"
    expected = "348088f33fb8080229f5475b95735719a43008a00af8a26b798831a138197eec"
    require(sha256(state) == receipt["members"]["requires-state.json"] == expected,
            "CAIMIRA state identity differs")
    require(sha256(PROVENANCE / "freeze.json") == receipt["freeze_sha256"] == refit["freeze_sha256"],
            "Frozen training manifest differs")
    require(sha256(PROVENANCE / "refit.json") == receipt["refit_sha256"],
            "Completed refit receipt differs")
    require(refit["checkpoint_sha256"] == receipt["checkpoint_sha256"],
            "Checkpoint identities differ")
    require(sha256(PROVENANCE / "combined-sources.json") == freeze["manifest"],
            "Source composition differs")
    require(sha256(PROVENANCE / "catalog-freeze-v6.json") == combined["catalog_receipt"],
            "Catalog freeze differs")
    require(len(combined["benchmarks"]) == 44 and len(freeze["folds"]) == 36
            and len(freeze["sources"]) == 176, "Corpus counts differ")

    with (ROOT / "data/training/projected_training_tables.csv").open(newline="") as handle:
        table_rows = list(csv.DictReader(handle))
    require({r["path"]: r["sha256"] for r in table_rows} == freeze["sources"],
            "Projected table inventory differs")
    with (ROOT / "data/local/pairs.csv").open(newline="") as handle:
        pairs = list(csv.DictReader(handle))
    pair_digest = hashlib.sha256("\x1f".join(p["pair_id"] for p in pairs).encode()).hexdigest()
    require(pair_digest == panel["pairs_sha256"], "Replay pair population differs")

    counts = Counter(p["benchmark"] for p in pairs)
    overlap = []
    for benchmark, count in sorted(counts.items()):
        if benchmark in combined["benchmarks"]:
            source = catalog["suites"][benchmark]["source"]
            hashes_match = all(source[t]["sha256"] == panel["sources"][benchmark][t + "_table"]["sha256"]
                               for t in ("items", "responses", "subjects"))
            require(hashes_match, f"Shared source table identity differs: {benchmark}")
            overlap.append({"benchmark": benchmark, "pairs": count})
    require(sum(row["pairs"] for row in overlap) == 25, "Shared-source count differs")
    result = {
        "status": "passed",
        "caimira_state_sha256": expected,
        "benchmarks": 44,
        "source_lineages": 36,
        "projected_tables": 176,
        "replay_pairs": len(pairs),
        "shared_source_pairs": 25,
        "shared_sources": overlap,
        "scope": "State-to-fit provenance and native source identity; exact evaluated-row intersections require the corpus and panel.",
        "rasch_historical_fit": "UNVERIFIED: original fit receipt remains a recovery target",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
