"""Resident CUDA fits and legally isolated replay of a frozen development matrix."""

# ruff: noqa: B023
# Loop-local job callbacks execute synchronously inside _job. They are never
# retained, queued or submitted concurrently; model/context residency depends
# on this single-writer execution contract.

from __future__ import annotations

import gc
import importlib.metadata
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from research.proxy_gpu.experiment import _stage
from research.proxy_gpu.models import pooling_probabilities
from research.proxy_gpu.runtime import EXPECTED_VERSIONS, check_resources, configure, resource_snapshot
from research.proxy_iteration.artifacts import canonical, digest, file_identity, read_json, write_once

from .data import load_panel_data, training_histories
from .features import load_features
from .guard import supervised_worker
from .protocol import ALC_WEIGHTS, BLOCK_LIMITS, BUDGETS, REPRESENTATIONS, SEEDS, STRUCTURED, load, structured_name

TABULAR_RUNTIME_PINS = {"9.2.0": {}, "2.2.1": {"transformers": "4.57.6"}}


class QualificationBlocked(RuntimeError):
    """A declared recipe failed its prospective CPU/CUDA release contract."""


class Budget:
    def __init__(self, root, worker):
        root = Path(root)
        self.root = root / "stages" / worker
        self.root.mkdir(parents=True, exist_ok=True)
        self.charges = defaultdict(float)
        for name in (
            "core-completion.json",
            "replication-completion.json",
            "tabular-9.2.0-completion.json",
            "tabular-2.2.1-completion.json",
        ):
            if (root / name).exists():
                for phase, value in read_json(root / name)["block_elapsed_seconds"].items():
                    self.charges[phase] += value
        self.previous_charges = dict(self.charges)
        self.sequence = 0
        self.started = time.monotonic()
        timing = read_json(root / "scientific-start.json")
        self.execution_started = timing["unix_seconds"]
        self.admission_seconds = timing["admission_seconds"]

    def call(self, phase, name, function, *args, **kwargs):
        available = BLOCK_LIMITS[phase] - self.charges[phase]
        global_remaining = 23400 - self.admission_seconds - (time.time() - self.execution_started)
        limit = min(1200, available, global_remaining)
        if limit <= 0:
            raise TimeoutError("frozen block/audit reserve exhausted: " + phase)
        self.sequence += 1
        stage = str(self.sequence) + ":" + name
        _stage(self.root, stage, limit)
        started = time.monotonic()
        try:
            result = function(*args, **kwargs)
            check_resources()
            return result
        finally:
            elapsed = time.monotonic() - started
            self.charges[phase] += elapsed
            _stage(self.root, stage, limit, active=False)


def targets_for(rows, traces):
    records = {r["response_id"]: r for r in rows}
    targets = []
    for trace in traces:
        for pair, selection in sorted(trace["pairs"].items()):
            for rid in selection["targets"]:
                row = records[rid]
                targets.append(
                    {
                        "run_id": trace["run_id"],
                        "pair_id": pair,
                        "response_id": rid,
                        "benchmark_id": row["benchmark_id"],
                        "subject_key": row["subject_key"],
                        "visible_input_id": digest([pair, row["item_key"]]),
                        "item_key": row["item_key"],
                        "source_group": row["group_id"],
                        "outcome": row["outcome"],
                    }
                )
    return targets


def _vectors(rows, matrix, index):
    return matrix[[index[r["item_key"]] for r in rows]]


def _cases(traces, fold_ids, records):
    for trace in traces:
        if trace["fold_id"] not in fold_ids:
            continue
        for pair, selection in sorted(trace["pairs"].items()):
            trials = [records[i] for i in selection["targets"]]
            unique = {r["item_key"]: r for r in trials}
            query = [unique[k] for k in sorted(unique)]
            for budget in BUDGETS:
                support = [records[i] for i in selection["support"][:budget]]
                if any(r["pair_id"] != pair for r in support + query):
                    raise ValueError("cross-pair evidence access")
                yield {
                    "run_id": trace["run_id"],
                    "pair_id": pair,
                    "budget": budget,
                    "support": support,
                    "query": query,
                    "trials": trials,
                }


def _emit(stream, case, candidate, probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if (
        probabilities.shape != (len(case["query"]),)
        or not np.isfinite(probabilities).all()
        or np.any((probabilities < 0) | (probabilities > 1))
    ):
        raise ArithmeticError("invalid probabilities: " + candidate)
    position = {r["item_key"]: i for i, r in enumerate(case["query"])}
    for row in case["trials"]:
        value = {
            "run_id": case["run_id"],
            "pair_id": case["pair_id"],
            "response_id": row["response_id"],
            "benchmark_id": row["benchmark_id"],
            "subject_key": row["subject_key"],
            "visible_input_id": digest([case["pair_id"], row["item_key"]]),
            "item_key": row["item_key"],
            "source_group": row["group_id"],
            "outcome": row["outcome"],
            "budget": case["budget"],
            "candidate_id": candidate,
            "prediction": float(probabilities[position[row["item_key"]]]),
            "fallback": None,
        }
        stream.write(canonical(value) + b"\n")


def _job(root, job_id, candidates, function, *, manifest_sha256, guard):
    directory = Path(root) / "jobs" / job_id
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status, failure, outputs = "COMPLETE", None, {}
    try:
        function(directory)
    except (ValueError, ArithmeticError, RuntimeError, TimeoutError, MemoryError, ImportError) as exc:
        status = "BLOCKED" if isinstance(exc, QualificationBlocked) else "FAILED"
        failure = {
            "type": type(exc).__name__,
            "message": str(exc)[:2000],
            "population_reduced": False,
            "automatic_retry": False,
        }
        # Partial files remain observable but never enter complete-job evidence.
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if (directory / "forecasts.jsonl").exists() and (directory / "forecasts.jsonl").stat().st_size:
            status = "PARTIAL"
    for path in directory.rglob("*"):
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            outputs[relative] = {"path": relative, **file_identity(path)}
    receipt = {
        "schema": "proxy-structured-job/v1",
        "id": job_id,
        "manifest_sha256": manifest_sha256,
        "status": status,
        "candidates": candidates,
        "failure": failure,
        "outputs": outputs,
        "elapsed_seconds": time.monotonic() - started,
        "resource": resource_snapshot(),
        "guard": guard,
    }
    write_once(directory / "receipt.json", receipt)
    return receipt


def _calibration_episodes(rows, train_ids, traces, matrix, index, *, matched):
    records = {r["response_id"]: r for r in rows if r["response_id"] in set(train_ids)}
    histories = training_histories(rows, train_ids, saved=traces if matched else None)
    episodes = []
    for trace in histories:
        for pair, selection in trace["pairs"].items():
            query = [records[i] for i in selection["targets"]]
            for budget, mass in zip(BUDGETS, ALC_WEIGHTS, strict=True):
                support = [records[i] for i in selection["support"][:budget]]
                probabilities = pooling_probabilities(
                    _vectors(query, matrix, index),
                    _vectors(support, matrix, index),
                    [r["outcome"] for r in support],
                    dtype="float64",
                )
                p = np.clip(probabilities, 1e-8, 1 - 1e-8)
                episodes.append(
                    {
                        "base_logits": np.log(p) - np.log1p(-p),
                        "y": np.array([r["outcome"] for r in query]),
                        "budget": budget,
                        "history_id": trace["run_id"],
                        "run_id": trace["run_id"],
                        "pair_id": pair,
                        "query_ids": selection["targets"],
                        "support_ids": selection["support"][:budget],
                        "weight": mass / len(histories) / len(trace["pairs"]),
                    }
                )
    return episodes


def _predict_structured(model, cases, matrices, index, subject_index, representation, base, seed, stream, directory):
    from .inference import fit_states, predict_states

    offsets, loadings, floors = model.item_parameters(matrices[representation])
    priors = model.prior(matrices["subjects"])
    requests, query_parameters, none = [], [], []
    evidence = []
    for case in cases:
        support = case["support"]
        s = [index[r["item_key"]] for r in support]
        q = [index[r["item_key"]] for r in case["query"]]
        mean = np.asarray(priors[subject_index[case["query"][0]["subject_key"]]], dtype=np.float64)
        dimension = len(mean)
        requests.append(
            {
                "offsets": offsets[s],
                "loadings": loadings[s],
                "floors": floors[s],
                "labels": [r["outcome"] for r in support],
                "prior_mean": mean,
                "prior_cov": np.eye(dimension),
                "pair_id": case["pair_id"],
                "evidence_pair_ids": [r["pair_id"] for r in support],
            }
        )
        query_parameters.append({"offsets": offsets[q], "loadings": loadings[q], "floors": floors[q]})
        logit = offsets[q] + loadings[q] @ mean
        none.append(floors[q] + (1 - floors[q]) / (1 + np.exp(-np.clip(logit, -700, 700))))
        evidence.append(
            {
                "run_id": case["run_id"],
                "pair_id": case["pair_id"],
                "budget": case["budget"],
                "support_ids": [r["response_id"] for r in support],
                "query_ids": [r["response_id"] for r in case["query"]],
            }
        )
    # Each update is an independent declared procedure. Preserve complete
    # no-adaptation predictions even when a later MAP/posterior branch fails.
    for i, case in enumerate(cases):
        _emit(stream, case, base + "_none.s" + str(seed), none[i])
    states = fit_states(requests, device="cuda", dtype="float64", max_seconds=1200)
    write_once(
        directory / "support-states.json",
        {"states": states, "evidence": evidence, "prediction_count_unique": sum(len(c["query"]) for c in cases)},
    )
    predictions = predict_states(states, query_parameters, device="cuda")
    for i, case in enumerate(cases):
        _emit(stream, case, base + "_map.s" + str(seed), predictions[i])
    posterior = representation == "minilm" and model.kind in {"rasch", "twopl", "threepl_global", "threepl_item"}
    posterior_predictions = None
    if posterior:
        posterior_states = fit_states(requests, device="cuda", dtype="float64", method="posterior", max_seconds=1200)
        write_once(directory / "posterior-states.json", {"states": posterior_states, "evidence": evidence})
        posterior_predictions = predict_states(posterior_states, query_parameters, device="cuda")
    if posterior:
        for i, case in enumerate(cases):
            _emit(stream, case, base + "_posterior.s" + str(seed), posterior_predictions[i])


def core_worker(root, expected_hash, *, replication=False):
    from .baselines import ess_pooling_probabilities, fit_calibration, fit_logistic
    from .structured import fit_structured

    root = Path(root)
    guard = supervised_worker(root / (("replication" if replication else "core") + "-exit.json"))
    load(root, expected_hash)
    configure()
    started = time.monotonic()
    budget = Budget(root, "replication" if replication else "core")
    receipts = []
    for track in ("matched", "broader"):
        rows, folds, traces, records = load_panel_data(root, track)
        matrices, index, subject_index = load_features(root, track)
        targets = targets_for(rows, traces)
        if (root / track / "targets.json").exists():
            if read_json(root / track / "targets.json") != targets:
                raise ValueError("primary/replication target populations differ")
        else:
            write_once(root / track / "targets.json", targets)
        all_cases = list(_cases(traces, {f["id"] for f in folds}, records))
        controls = ["pm8", "fixed_pool", "ess_pool"] + (["archival_lexical"] if track == "matched" else [])

        def replay_controls(directory):
            archived = {}
            if track == "matched":
                from .data import jsonl

                archived = {
                    (r["run_id"], r["pair_id"], r["response_id"], r["budget"]): r["prediction"]
                    for r in jsonl(root / track / "inputs/archival-lexical.jsonl")
                }
            with (directory / "forecasts.jsonl").open("xb") as stream:
                for case in all_cases:
                    labels = [r["outcome"] for r in case["support"]]
                    q, s = (_vectors(case[key], matrices["pool"], index) for key in ("query", "support"))
                    _emit(stream, case, "pm8", np.full(len(q), (4 + sum(labels)) / (8 + len(labels))))
                    _emit(stream, case, "fixed_pool", pooling_probabilities(q, s, labels, dtype="float64"))
                    _emit(stream, case, "ess_pool", ess_pooling_probabilities(q, s, labels))
                    if track == "matched":
                        _emit(
                            stream,
                            case,
                            "archival_lexical",
                            [
                                archived[case["run_id"], case["pair_id"], r["response_id"], case["budget"]]
                                for r in case["query"]
                            ],
                        )

        if not replication:
            receipts.append(
                _job(
                    root,
                    track + "-controls",
                    controls,
                    lambda d: budget.call("controls", track + "-controls", replay_controls, d),
                    manifest_sha256=expected_hash,
                    guard=guard,
                )
            )
        if track == "broader" and not replication:

            def legacy_job(directory):
                from research.proxy_iteration.artifacts import copy_once

                copy_once(root / "broader/inputs/legacy-forecasts.jsonl", directory, "forecasts.jsonl")
                copy_once(root / "broader/inputs/legacy-reuse.json", directory, "legacy-reuse.json")

            receipts.append(
                _job(root, "broader-legacy", ["legacy_irt"], legacy_job, manifest_sha256=expected_hash, guard=guard)
            )
        banks = defaultdict(list)
        for fold in folds:
            banks[digest(sorted(fold["train_response_ids"]))].append(fold)
        for bank, matching_folds in banks.items():
            train_ids = sorted(matching_folds[0]["train_response_ids"])
            train = [records[i] for i in train_ids]
            cases = list(_cases(traces, {f["id"] for f in matching_folds}, records))
            labels, pairs = [r["outcome"] for r in train], [r["pair_id"] for r in train]
            z = matrices["subjects"][[subject_index[r["subject_key"]] for r in train]]
            for seed in SEEDS:
                baseline_recipes = (
                    ("lexical", "lexical_head_tail_wordpiece", "equal_pair", 0.01),
                    ("minilm_logistic", "minilm", "equal_pair", 0.01),
                    ("minilm_ridge01", "minilm", "equal_pair", 0.1),
                    ("minilm_uniform", "minilm", "uniform_response", 0.01),
                )
                for name, representation, weighting, ridge in () if replication else baseline_recipes:
                    candidate = f"{name}.s{seed}"

                    def logistic_job(directory):
                        artifact, model = fit_logistic(
                            _vectors(train, matrices[representation], index),
                            labels,
                            pairs,
                            seed=seed,
                            weighting=weighting,
                            ridge=ridge,
                        )
                        write_once(directory / "model.json", artifact)
                        write_once(
                            directory / "lineage.json",
                            {"training_ids": train_ids, "folds": [f["id"] for f in matching_folds]},
                        )
                        with (directory / "forecasts.jsonl").open("xb") as stream:
                            for case in cases:
                                _emit(
                                    stream,
                                    case,
                                    candidate,
                                    model.predict(_vectors(case["query"], matrices[representation], index)),
                                )

                    receipts.append(
                        _job(
                            root,
                            f"{track}-{bank[:12]}-{candidate}",
                            [candidate],
                            lambda d: budget.call("controls", candidate, logistic_job, d),
                            manifest_sha256=expected_hash,
                            guard=guard,
                        )
                    )
                candidate = f"pool_calibrated.s{seed}"

                def calibration_job(directory):
                    episodes = _calibration_episodes(
                        rows, train_ids, traces, matrices["pool"], index, matched=track == "matched"
                    )
                    artifact, model = fit_calibration(episodes, seed=seed, training_ids=train_ids)
                    write_once(directory / "model.json", artifact)
                    write_once(
                        directory / "episode-lineage.json",
                        [
                            {k: e[k] for k in ("run_id", "pair_id", "budget", "support_ids", "query_ids", "weight")}
                            for e in episodes
                        ],
                    )
                    with (directory / "forecasts.jsonl").open("xb") as stream:
                        for case in cases:
                            p = pooling_probabilities(
                                _vectors(case["query"], matrices["pool"], index),
                                _vectors(case["support"], matrices["pool"], index),
                                [r["outcome"] for r in case["support"]],
                                dtype="float64",
                            )
                            p = np.clip(p, 1e-8, 1 - 1e-8)
                            _emit(stream, case, candidate, model.predict(np.log(p) - np.log1p(-p), case["budget"]))

                if not replication:
                    receipts.append(
                        _job(
                            root,
                            f"{track}-{bank[:12]}-{candidate}",
                            [candidate],
                            lambda d: budget.call("controls", candidate, calibration_job, d),
                            manifest_sha256=expected_hash,
                            guard=guard,
                        )
                    )
                for representation in REPRESENTATIONS[track][1:] if replication else ("minilm",):
                    for kind, prior in STRUCTURED:
                        name = structured_name(representation, kind, prior)
                        candidates = [f"{name}_{update}.s{seed}" for update in ("none", "map")]
                        if representation == "minilm" and kind in {"rasch", "twopl", "threepl_global", "threepl_item"}:
                            candidates.append(f"{name}_posterior.s{seed}")

                        def structured_job(directory):
                            model, artifact = fit_structured(
                                _vectors(train, matrices[representation], index),
                                labels,
                                pairs,
                                [r["subject_key"] for r in train],
                                [r["item_key"] for r in train],
                                z,
                                kind=kind,
                                prior_kind=prior,
                                rank=2,
                                seed=seed,
                            )
                            write_once(directory / "model.json", model.export())
                            write_once(directory / "fit.json", artifact)
                            write_once(
                                directory / "lineage.json",
                                {"training_ids": train_ids, "folds": [f["id"] for f in matching_folds]},
                            )
                            with (directory / "forecasts.jsonl").open("xb") as stream:
                                _predict_structured(
                                    model,
                                    cases,
                                    matrices,
                                    index,
                                    subject_index,
                                    representation,
                                    name,
                                    seed,
                                    stream,
                                    directory,
                                )

                        phase = (
                            "adaptation_replication"
                            if representation != "minilm"
                            else "multidimensional"
                            if kind in {"mirt", "caimira"}
                            else "scalar_tabular"
                        )
                        receipts.append(
                            _job(
                                root,
                                f"{track}-{bank[:12]}-{name}.s{seed}",
                                candidates,
                                lambda d: budget.call(phase, name, structured_job, d),
                                manifest_sha256=expected_hash,
                                guard=guard,
                            )
                        )
            gc.collect()
    summary = {
        "schema": "proxy-structured-core/v1",
        "status": "COMPLETE_WITH_RECORDED_FAILURES" if any(r["status"] != "COMPLETE" for r in receipts) else "COMPLETE",
        "job_count": len(receipts),
        "complete_jobs": sum(r["status"] == "COMPLETE" for r in receipts),
        "failed_jobs": sum(r["status"] != "COMPLETE" for r in receipts),
        "elapsed_seconds": time.monotonic() - started,
        "block_elapsed_seconds": {k: v - budget.previous_charges.get(k, 0) for k, v in budget.charges.items()},
        "resource": resource_snapshot(),
        "manifest_sha256": expected_hash,
        "population_reductions": 0,
    }
    load(root, expected_hash)
    write_once(root / ("replication-completion.json" if replication else "core-completion.json"), summary)
    return summary


def tabular_runtime(version):
    """Configure an optional-runtime phase and enforce that environment's qualified pins."""
    identity = dict(configure(strict_pins=False)["identity"])
    expected = {**EXPECTED_VERSIONS, **TABULAR_RUNTIME_PINS[version], "tabpfn": version}
    identity.update(tabpfn=importlib.metadata.version("tabpfn"), expected_versions=expected)
    identity["pin_mismatches"] = {
        name: {"expected": value, "actual": identity.get(name)}
        for name, value in expected.items()
        if identity.get(name) != value
    }
    if identity["pin_mismatches"]:
        raise RuntimeError(f"runtime pins do not match: {identity['pin_mismatches']}")
    return identity


def tabular_worker(root, expected_hash, checkpoint, version):
    """A sequential optional-runtime phase; the qualified core worker has exited."""
    from .baselines import fit_boosted
    from .inference import fit_states, predict_states
    from .tabular import fit_tabpfn, probabilities_to_logits

    if version not in ("2.2.1", "9.2.0"):
        raise ValueError("exact supported TabPFN version required")
    root = Path(root)
    guard = supervised_worker(root / (("tabular920" if version == "9.2.0" else "tabular221") + "-exit.json"))
    manifest = load(root, expected_hash)
    runtime = tabular_runtime(version)
    budget = Budget(root, "tabular920" if version == "9.2.0" else "tabular221")
    receipts = []
    for track in ("matched", "broader"):
        rows, folds, traces, records = load_panel_data(root, track)
        matrices, index, _ = load_features(root, track)
        banks = defaultdict(list)
        for fold in folds:
            banks[digest(sorted(fold["train_response_ids"]))].append(fold)
        for bank, matching_folds in banks.items():
            train_ids = sorted(matching_folds[0]["train_response_ids"])
            train = [records[i] for i in train_ids]
            x, y, pairs = (
                _vectors(train, matrices["minilm"], index),
                [r["outcome"] for r in train],
                [r["pair_id"] for r in train],
            )
            cases = list(_cases(traces, {f["id"] for f in matching_folds}, records))
            for seed in SEEDS:
                tag = "920" if version == "9.2.0" else "221"
                candidates = [f"tabpfn{tag}_{update}.s{seed}" for update in ("none", "map")]

                def tabpfn_job(directory):
                    if any(c.rsplit(".s", 1)[0] in manifest["blocked_procedures"] for c in candidates):
                        raise QualificationBlocked(
                            "TabPFN procedure failed its pinned numerical qualification; no scientific fit attempted"
                        )
                    artifact, model = fit_tabpfn(
                        x, y, pairs, seed=seed, checkpoint=checkpoint, expected_version=version
                    )
                    write_once(
                        directory / "lineage.json",
                        {"training_ids": train_ids, "folds": [f["id"] for f in matching_folds]},
                    )
                    p = model.predict(matrices["minilm"])
                    write_once(directory / "model.json", artifact)
                    base_logits = probabilities_to_logits(p)
                    requests, query_parameters, evidence = [], [], []
                    for case in cases:
                        s = [index[r["item_key"]] for r in case["support"]]
                        q = [index[r["item_key"]] for r in case["query"]]
                        requests.append(
                            {
                                "offsets": base_logits[s],
                                "loadings": np.ones((len(s), 1)),
                                "floors": np.zeros(len(s)),
                                "labels": [r["outcome"] for r in case["support"]],
                                "prior_mean": [0.0],
                                "prior_cov": [[1.0]],
                                "pair_id": case["pair_id"],
                                "evidence_pair_ids": [r["pair_id"] for r in case["support"]],
                            }
                        )
                        query_parameters.append(
                            {"offsets": base_logits[q], "loadings": np.ones((len(q), 1)), "floors": np.zeros(len(q))}
                        )
                        evidence.append(
                            {
                                "run_id": case["run_id"],
                                "pair_id": case["pair_id"],
                                "budget": case["budget"],
                                "support_ids": [r["response_id"] for r in case["support"]],
                            }
                        )
                    with (directory / "forecasts.jsonl").open("xb") as stream:
                        for case in cases:
                            q = [index[r["item_key"]] for r in case["query"]]
                            _emit(stream, case, candidates[0], p[q])
                        stream.flush()
                        states = fit_states(requests, device="cuda", dtype="float64", max_seconds=1200)
                        write_once(directory / "support-states.json", {"states": states, "evidence": evidence})
                        mapped = predict_states(states, query_parameters, device="cuda")
                        for case, adapted in zip(cases, mapped, strict=True):
                            _emit(stream, case, candidates[1], adapted)

                receipts.append(
                    _job(
                        root,
                        f"{track}-{bank[:12]}-tabpfn{tag}.s{seed}",
                        candidates,
                        lambda d: budget.call("scalar_tabular", candidates[0], tabpfn_job, d),
                        manifest_sha256=expected_hash,
                        guard=guard,
                    )
                )
                if version == "9.2.0":
                    for recipe in ("shallow", "regularized"):
                        candidate = f"boost_{recipe}.s{seed}"

                        def boosted_job(directory):
                            if candidate.rsplit(".s", 1)[0] in manifest["blocked_procedures"]:
                                raise QualificationBlocked(
                                    "boosting procedure failed its pinned numerical qualification; no scientific fit attempted"
                                )
                            artifact, model = fit_boosted(x, y, pairs, seed=seed, recipe=recipe)
                            write_once(directory / "model.json", artifact)
                            write_once(
                                directory / "lineage.json",
                                {"training_ids": train_ids, "folds": [f["id"] for f in matching_folds]},
                            )
                            probabilities = model.predict(matrices["minilm"])
                            with (directory / "forecasts.jsonl").open("xb") as stream:
                                for case in cases:
                                    _emit(
                                        stream,
                                        case,
                                        candidate,
                                        probabilities[[index[r["item_key"]] for r in case["query"]]],
                                    )

                        receipts.append(
                            _job(
                                root,
                                f"{track}-{bank[:12]}-{candidate}",
                                [candidate],
                                lambda d: budget.call("controls", candidate, boosted_job, d),
                                manifest_sha256=expected_hash,
                                guard=guard,
                            )
                        )
    summary = {
        "schema": "proxy-structured-tabular-phase/v1",
        "version": version,
        "status": "COMPLETE_WITH_RECORDED_FAILURES" if any(r["status"] != "COMPLETE" for r in receipts) else "COMPLETE",
        "jobs": len(receipts),
        "failed_jobs": sum(r["status"] == "FAILED" for r in receipts),
        "block_elapsed_seconds": {k: v - budget.previous_charges.get(k, 0) for k, v in budget.charges.items()},
        "resource": resource_snapshot(),
        "runtime_identity": runtime,
        "manifest_sha256": expected_hash,
    }
    load(root, expected_hash)
    write_once(root / ("tabular-" + version + "-completion.json"), summary)
    return summary
