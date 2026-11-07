"""Check exact candidate inference with a fixed synthetic fixture."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import scipy
import torch

from predict_cli import load_candidate, verify_candidate


ROOT = Path(__file__).resolve().parent
BUDGETS = (0, 1, 3, 7, 15, 31)


def ensure(condition: bool, message: str) -> None:
    if condition is False:
        raise AssertionError(message)


def checked_probability(module, query, support):
    value = module.predict(query, labeled=support)
    ensure(type(value) is float, "Predictor output must be a native Python float")
    ensure(math.isfinite(value) and 0.0 <= value <= 1.0, "Predictor output must be finite and bounded")
    return value


def expect_value_error(module, query, support):
    try:
        module.predict(query, labeled=support)
    except ValueError as error:
        return {"type": type(error).__name__, "message": str(error)}
    raise AssertionError("Expected ValueError was absent")


def main() -> None:
    started = time.perf_counter()
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    fixture = json.loads((ROOT / "fixtures/fixture.json").read_text())
    query, support = fixture["input"], fixture["labeled"]
    before = verify_candidate(ROOT / "candidate")
    module = load_candidate(ROOT / "candidate")

    predictions = {str(n): checked_probability(module, query, support[:n]) for n in BUDGETS}
    repeated = {str(n): checked_probability(module, query, support[:n]) for n in BUDGETS}
    ensure(predictions == repeated, "Repeated calls changed the fixture outputs")

    unrelated_subject = dict(query[0], model_name="unrelated-synthetic-system")
    other_benchmark = dict(support[0][0][1], benchmark_id="different-synthetic-benchmark")
    distractors = [((unrelated_subject, support[0][0][1]), 1), ((query[0], other_benchmark), 0)]
    isolated = checked_probability(module, query, support[:7] + distractors)
    ensure(isolated == predictions["7"], "Ineligible support changed a prediction")

    missing_benchmark = [query[0], {key: value for key, value in query[1].items() if key != "benchmark_id"}]
    fallback = checked_probability(module, missing_benchmark, support[:7])
    ensure(fallback == predictions["0"], "Missing benchmark changed the frozen-state forecast")

    guards = {
        "32_eligible_labels": expect_value_error(module, query, support[:32]),
        "boolean_label": expect_value_error(module, query, [(support[0][0], True)]),
        "label_outside_binary_range": expect_value_error(module, query, [(support[0][0], 2)]),
    }

    command = [sys.executable, str(ROOT / "predict_cli.py"), "--input", str(ROOT / "fixtures/budgets.jsonl")]
    process_runs = []
    for name in ("first", "second"):
        completed = subprocess.run(command, text=True, capture_output=True, timeout=90)
        (results_dir / f"cli_{name}.jsonl").write_text(completed.stdout)
        (results_dir / f"cli_{name}.stderr.txt").write_text(completed.stderr)
        ensure(completed.returncode == 0, f"Fresh-process CLI failed: {completed.stderr}")
        rows = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
        observed = {str(row["id"]): row["probability"] for row in rows}
        ensure(observed == predictions, "Fresh-process predictions differ from the in-process fixture")
        process_runs.append({"command": command, "exit_code": completed.returncode, "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest()})
    ensure(process_runs[0]["stdout_sha256"] == process_runs[1]["stdout_sha256"], "Fresh-process output files differ")
    ensure(before == verify_candidate(ROOT / "candidate"), "Candidate files changed during inference")

    report = {
        "status": "passed",
        "scope": "Synthetic inference and interface checks; zero accuracy estimates or offline fitting runs",
        "candidate_sha256": "23546eb90c7469fe7ea2704879775885f1b32e3277b4b7b3a1015aa1c7c6297b",
        "candidate_members": before,
        "fixture_sha256": hashlib.sha256((ROOT / "fixtures/fixture.json").read_bytes()).hexdigest(),
        "label_budgets": list(BUDGETS),
        "probabilities_by_budget": predictions,
        "checks": {
            "native_float_finite_unit_interval": True,
            "same_process_exact_repeat": True,
            "two_fresh_process_exact_repeats": True,
            "ineligible_subject_and_benchmark_support_ignored": True,
            "missing_benchmark_uses_frozen_prediction": True,
            "32_label_guard": True,
            "boolean_label_guard": True,
            "binary_label_guard": True,
            "candidate_files_unchanged": True,
        },
        "guard_errors": guards,
        "fresh_process_runs": process_runs,
        "environment": {
            "python": platform.python_version(),
            "executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "torch_threads": torch.get_num_threads(),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    (results_dir / "smoke_results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
