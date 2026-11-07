"""Run the exact candidate on JSON Lines requests using its frozen states."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any


EXPECTED_FILES = {
    "model.py": "a246beb1d96745d00094483bf86ef0ed585cc713dc46c805f15fd445b19dc21e",
    "caimira-requires-state.json": "348088f33fb8080229f5475b95735719a43008a00af8a26b798831a138197eec",
    "rasch-requires-state.json": "477130aaf555b872320f320c69458d91c38f877024163b10f5f887d2fcbc1ef7",
}


def default_candidate_dir() -> Path:
    here = Path(__file__).resolve().parent
    if (here / "model.py").is_file():
        return here
    return here / "candidate"


def verify_candidate(directory: Path) -> dict[str, str]:
    observed = {}
    for name, expected in EXPECTED_FILES.items():
        path = directory / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"Candidate integrity mismatch: {name}")
        observed[name] = digest
    return observed


def load_candidate(directory: Path | None = None):
    directory = (directory or default_candidate_dir()).resolve()
    verify_candidate(directory)
    spec = importlib.util.spec_from_file_location("paiec_candidate18", directory / "model.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Candidate module loader is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if callable(getattr(module, "predict", None)) is False:
        raise RuntimeError("Candidate predict entry point is unavailable")
    return module


def validate_pair(pair: Any, name: str) -> None:
    if isinstance(pair, (list, tuple)) and len(pair) == 2:
        subject, item = pair
    else:
        raise ValueError(f"{name} must be [subject, item]")
    if isinstance(subject, dict) and isinstance(item, dict):
        pass
    else:
        raise ValueError(f"{name} requires subject and item dictionaries")
    json.dumps(subject, allow_nan=False, sort_keys=True)
    if isinstance(item.get("item_content"), str) is False:
        raise ValueError(f"{name}.item_content must be a string")
    benchmark = item.get("benchmark_id")
    if benchmark is None or isinstance(benchmark, str):
        pass
    else:
        raise ValueError(f"{name}.benchmark_id must be a string when supplied")


def predict_request(module: Any, request: Any) -> dict[str, Any]:
    if isinstance(request, dict) is False:
        raise ValueError("Each request must be a JSON object")
    query = request.get("input")
    validate_pair(query, "input")
    support = request.get("labeled", [])
    if support is None:
        support = []
    if isinstance(support, list) is False:
        raise ValueError("labeled must be a list")
    for index, entry in enumerate(support):
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            pair, label = entry
        else:
            raise ValueError(f"labeled[{index}] must be [[subject, item], label]")
        validate_pair(pair, f"labeled[{index}]")
        if type(label) is int and label in (0, 1):
            pass
        else:
            raise ValueError(f"labeled[{index}] requires an integer 0 or 1")
    probability = module.predict(query, labeled=support)
    if type(probability) is float and math.isfinite(probability) and 0.0 <= probability <= 1.0:
        result = {"probability": probability}
    else:
        raise ArithmeticError("Candidate returned an invalid probability")
    if "id" in request:
        result["id"] = request["id"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, default=default_candidate_dir())
    parser.add_argument("--input", type=Path, help="JSON Lines file; default reads standard input")
    args = parser.parse_args()
    module = load_candidate(args.candidate_dir)
    stream = args.input.open(encoding="utf-8") if args.input else nullcontext(sys.stdin)
    with stream as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    result = predict_request(module, json.loads(line))
                except Exception as error:
                    raise ValueError(f"Request line {number}: {type(error).__name__}: {error}") from error
                print(json.dumps(result, allow_nan=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2)
