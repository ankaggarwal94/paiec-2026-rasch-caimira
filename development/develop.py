#!/usr/bin/env python3
"""Portable JSONL adapter around the preserved PAIEC structural fitter."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
ADAPTER_SCHEMA = "paiec-normalized-jsonl-adapter-v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def verify_source():
    manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        path = ROOT / "historical_source" / entry["path"]
        if path.is_file() is False or path.is_symlink():
            raise ValueError("frozen source missing or symlinked: " + entry["path"])
        if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            raise ValueError("frozen source differs: " + entry["path"])
    return len(manifest["files"])


def runtime(threads=1):
    verify_source()
    if type(threads) != int or threads < 1:
        raise ValueError("threads must be a positive integer")
    sys.path.insert(0, str(ROOT / "historical_source"))
    import torch
    from research.broad_transfer import predictors
    torch.set_num_threads(threads)
    return torch, predictors


def versions(torch):
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": importlib.metadata.version("numpy"),
        "scipy": importlib.metadata.version("scipy"),
        "device": "cpu",
        "threads": torch.get_num_threads(),
    }


def read_rows(path, *, training, max_records=100000):
    """Validate each trial; consolidate repeated visible input identities."""
    if max_records < 1:
        raise ValueError("max-records must be positive")
    allowed = {"benchmark", "subject", "item", "group", "text"}
    if training:
        allowed.add("label")
    rows, indices, labels, lookup, original = [], [], [], {}, []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip() == "":
                continue
            if len(original) >= max_records:
                raise ValueError("record limit exceeded; raise --max-records explicitly")
            if len(line.encode("utf-8")) > 64 * 1024**2:
                raise ValueError(f"line {line_number} exceeds 64 MiB")
            value = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
            if type(value) != dict or set(value) != allowed:
                raise ValueError(f"line {line_number}: exact required fields are {sorted(allowed)}")
            for field in ("benchmark", "subject", "item", "group", "text"):
                if type(value[field]) != str:
                    raise ValueError(f"line {line_number}: {field} must be a string")
            for field in ("benchmark", "subject", "item", "group"):
                if value[field] == "":
                    raise ValueError(f"line {line_number}: {field} must be nonempty")
            if training and (type(value["label"]) != int or (value["label"] in (0, 1)) is False):
                raise ValueError(f"line {line_number}: label must be integer 0 or 1")
            row = {field: value[field] for field in ("benchmark", "subject", "item", "group", "text")}
            key = row["benchmark"], row["subject"], row["item"]
            index = lookup.get(key)
            if index is None:
                index = len(rows)
                rows.append(row)
                lookup[key] = index
            elif rows[index] != row:
                raise ValueError(f"line {line_number}: repeated input identity has conflicting text or group")
            indices.append(index)
            original.append(row)
            if training:
                labels.append(value["label"])
    if len(rows) == 0:
        raise ValueError("input requires at least one record")
    return rows, indices, labels, original


def fit(args):
    output = Path(args.output)
    if output.exists():
        raise ValueError("output already exists; choose a fresh output directory")
    rows, indices, labels, _ = read_rows(args.input, training=True, max_records=args.max_records)
    torch, predictors = runtime(args.threads)
    bank = predictors.FeatureBank(rows)
    train = predictors.Training(bank, indices, labels)
    output.mkdir(parents=True)
    methods = ("rasch", "caimira") if args.method == "both" else (args.method,)
    common = {
        "schema": ADAPTER_SCHEMA,
        "adapter_sha256": digest(__file__),
        "source_manifest_sha256": digest(ROOT / "SOURCE_MANIFEST.json"),
        "input_sha256": digest(args.input),
        "normalization": "exact supplied subject and text strings; repeated benchmark/subject/item identities consolidated",
        "record_count": len(labels),
        "distinct_inputs": len(rows),
        "distinct_subject_strings": len(set(row["subject"] for row in rows)),
        "pairs": train.pairs,
        "total_pair_weight": float(train.weight.sum()),
        "runtime": versions(torch),
        "execution_scope": "new fit on supplied normalized trials; source-corpus projection is external",
    }
    receipts = {}
    for method in methods:
        predictor = predictors.fit_predictor(method, bank, indices, labels, seed=args.seed, device="cpu")
        state_path = output / f"{method}-state.json"
        checkpoint_path = output / f"{method}-predictor.pt"
        write_json(state_path, predictor.state)
        predictor.save(checkpoint_path)
        receipts[method] = {
            **common,
            "fit": predictor.receipt,
            "state_sha256": digest(state_path),
            "checkpoint_sha256": digest(checkpoint_path),
        }
        write_json(output / f"{method}-receipt.json", receipts[method])
    print(json.dumps({"output": str(output), "fits": {key: value["fit"] for key, value in receipts.items()}}, allow_nan=False))


def predict(args):
    rows, indices, _, original = read_rows(args.input, training=False, max_records=args.max_records)
    _, predictors = runtime(args.threads)
    state = json.loads(Path(args.state).read_text(encoding="utf-8"))
    method = state["config"]["kind"]
    if (method in ("rasch", "caimira")) is False:
        raise ValueError("state kind must be rasch or caimira")
    model = predictors.Predictor(method, state, {}, "cpu")
    probabilities = model.predict(predictors.FeatureBank(rows), indices)
    for row, probability in zip(original, probabilities, strict=True):
        value = float(probability)
        if math.isfinite(value) is False or value < 0 or value > 1:
            raise ArithmeticError("prediction requires a finite unit probability")
        print(json.dumps({"benchmark": row["benchmark"], "item": row["item"], "subject": row["subject"], "probability": value}, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify-source", help="verify preserved source bytes")
    fit_parser = sub.add_parser("fit", help="fit frozen Rasch or CAIMIRA recipe to normalized JSONL")
    fit_parser.add_argument("--input", required=True)
    fit_parser.add_argument("--output", required=True)
    fit_parser.add_argument("--method", choices=("rasch", "caimira", "both"), default="both")
    fit_parser.add_argument("--seed", type=int, default=17)
    fit_parser.add_argument("--threads", type=int, default=1)
    fit_parser.add_argument("--max-records", type=int, default=100000)
    predict_parser = sub.add_parser("predict", help="predict from an exported structural state with zero purchased labels")
    predict_parser.add_argument("--state", required=True)
    predict_parser.add_argument("--input", required=True)
    predict_parser.add_argument("--threads", type=int, default=1)
    predict_parser.add_argument("--max-records", type=int, default=100000)
    args = parser.parse_args()
    try:
        if args.command == "verify-source":
            print(json.dumps({"source_files_verified": verify_source()}))
        elif args.command == "fit":
            fit(args)
        else:
            predict(args)
    except (ValueError, KeyError, OSError, ArithmeticError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
