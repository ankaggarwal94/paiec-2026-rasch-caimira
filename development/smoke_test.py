#!/usr/bin/env python3
"""Exercise native fitting and learned-state export using synthetic trials only."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

import develop

ROOT = Path(__file__).resolve().parent


def require(condition, message):
    if bool(condition) is False:
        raise AssertionError(message)


def run(*arguments):
    result = subprocess.run([sys.executable, str(ROOT / "develop.py"), *map(str, arguments)], capture_output=True, text=True)
    require(result.returncode == 0, result.stderr)
    return result.stdout


def main():
    output = ROOT / "smoke-results"
    output.mkdir(exist_ok=True)
    torch, predictors = develop.runtime(1)
    import numpy as np
    from research.proxy_gpu.models import _parameters, _torch
    from research.proxy_structured import structured as irt

    training = ROOT / "fixtures" / "synthetic-training.jsonl"
    query = ROOT / "fixtures" / "synthetic-queries.jsonl"
    rows, indices, labels, _ = develop.read_rows(training, training=True)
    query_rows, query_indices, _, _ = develop.read_rows(query, training=False)
    bank, query_bank = predictors.FeatureBank(rows), predictors.FeatureBank(query_rows)
    statistics = predictors.Training(bank, indices, labels)
    pair_weights = np.bincount(statistics.pair, weights=statistics.weight)
    require(np.allclose(pair_weights, 1 / statistics.pairs, rtol=0, atol=1e-15), "pair totals differ")

    with tempfile.TemporaryDirectory(prefix="paiec-development-smoke-") as temp:
        temporary = Path(temp)
        fit_dir = temporary / "fits"
        fit_stdout = run("fit", "--input", training, "--output", fit_dir, "--method", "both", "--seed", 17)
        (output / "fit-stdout.json").write_text(fit_stdout, encoding="utf-8")
        results = {}
        for method in ("rasch", "caimira"):
            state_path = fit_dir / f"{method}-state.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            receipt = json.loads((fit_dir / f"{method}-receipt.json").read_text(encoding="utf-8"))
            restored_json = predictors.Predictor(method, state, {}, "cpu")
            restored_pt = predictors.load_predictor(fit_dir / f"{method}-predictor.pt", device="cpu")
            p_json = restored_json.predict(query_bank, query_indices)
            p_pt = restored_pt.predict(query_bank, query_indices)
            require(np.array_equal(p_json, p_pt), "JSON and checkpoint predictions differ")
            require(np.isfinite(p_json).all() and (p_json >= 0).all() and (p_json <= 1).all(), "invalid probability")
            require(("nuisance" in state["parameters"]) is False, "training subject nuisance leaked into state")
            require(state["config"]["dimension"] == state["config"]["subject_dimension"] == 384, "feature width differs")
            require(state["config"]["rank"] == (1 if method == "rasch" else 2), "rank differs")

            library, dtype = _torch("cpu", "float32")
            initial = _parameters(library, irt._shapes(state["config"], len(bank.z)), 17, "cpu", dtype)
            with torch.no_grad():
                initial["intercept"].zero_()
                if method == "caimira":
                    initial["relevance_bias"].zero_()
            parameter_change = max(float(np.max(np.abs(np.asarray(value) - initial[name].detach().cpu().numpy()))) for name, value in state["parameters"].items())
            require(parameter_change > 1e-5, "exported state still equals initialization")

            repeated = predictors.fit_predictor(method, bank, indices, labels, seed=17, device="cpu")
            require(repeated.state == state, "same-seed repeated fit differs")
            prediction_stdout = run("predict", "--state", state_path, "--input", query)
            fresh = np.array([json.loads(line)["probability"] for line in prediction_stdout.splitlines()])
            require(np.array_equal(fresh, p_json), "fresh process prediction differs")
            (output / f"{method}-predictions.jsonl").write_text(prediction_stdout, encoding="utf-8")
            develop.write_json(output / f"{method}-state.json", state)
            develop.write_json(output / f"{method}-receipt.json", receipt)
            results[method] = {
                "json_checkpoint_prediction_match": True,
                "fresh_process_prediction_match": True,
                "same_seed_state_match": True,
                "maximum_parameter_change_from_initialization": parameter_change,
                "probability_min": float(p_json.min()),
                "probability_max": float(p_json.max()),
                "diagnostics": receipt["fit"]["diagnostics"],
            }

        first = json.loads(training.read_text(encoding="utf-8").splitlines()[0])
        rejected = []
        for name, changed in (("boolean_label", {**first, "label": True}), ("nonbinary_label", {**first, "label": 2}), ("subject_object", {**first, "subject": {"name": "toy"}})):
            path = temporary / f"{name}.jsonl"
            path.write_text(json.dumps(changed) + "\n", encoding="utf-8")
            try:
                develop.read_rows(path, training=True)
            except ValueError:
                rejected.append(name)
            else:
                raise AssertionError("invalid record accepted: " + name)
        conflicting = temporary / "conflict.jsonl"
        conflicting.write_text(json.dumps(first) + "\n" + json.dumps({**first, "text": "changed content"}) + "\n", encoding="utf-8")
        try:
            develop.read_rows(conflicting, training=True)
        except ValueError:
            rejected.append("conflicting_input_identity")
        else:
            raise AssertionError("conflicting input accepted")

    summary = {
        "scope": "synthetic functional training and serialization checks only",
        "source_files_verified": develop.verify_source(),
        "runtime": develop.versions(torch),
        "training_trials": len(labels),
        "distinct_inputs": len(rows),
        "pairs": statistics.pairs,
        "equal_pair_weight_check": True,
        "pair_weights": pair_weights.tolist(),
        "rejected_inputs": rejected,
        "fits": results,
        "status": "PASS",
    }
    develop.write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
