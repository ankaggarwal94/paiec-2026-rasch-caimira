"""Admit existing development evidence without changing its trials or histories.

All persisted paths are relative to the caller-owned evidence root. Cached
arrays are copied only after their original completion bindings are verified.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from research.proxy_iteration.artifacts import canonical, copy_once, digest, file_identity, read_json, write_once

BUDGETS = (0, 1, 3, 7, 15, 31)
HISTORIES = (20261030, 20261031, 20261032)


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_bytes().splitlines() if line.strip()]


def _write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        for row in rows:
            handle.write(canonical(row) + b"\n")
    return file_identity(path)


def _valid_rows(rows):
    seen, visible, subjects = set(), {}, {}
    required = {
        "response_id",
        "benchmark_id",
        "pair_id",
        "subject_key",
        "item_key",
        "group_id",
        "outcome",
        "subject",
        "item",
    }
    for row in rows:
        if not required <= set(row) or row["response_id"] in seen:
            raise ValueError("missing identity or repeated response ID")
        if type(row["outcome"]) not in (int, float) or row["outcome"] not in (0, 1):
            raise ValueError("strict numeric binary outcomes required")
        for key in required - {"outcome", "subject", "item"}:
            if not isinstance(row[key], str) or not row[key]:
                raise ValueError("nonempty response/subject/item/group identities required")
        for key in ("subject", "item"):
            if not isinstance(row[key], dict) or any(type(v) is not str for v in row[key].values()):
                raise ValueError("published visible fields must be strings")
        value = digest([row["subject"], row["item"]])
        if subjects.setdefault(row["subject_key"], digest(row["subject"])) != digest(row["subject"]):
            raise ValueError("one subject key maps to changing full configurations")
        if visible.setdefault((row["subject_key"], row["item_key"]), value) != value:
            raise ValueError("conflicting visible identities")
        seen.add(row["response_id"])
    return {r["response_id"]: r for r in rows}


def validate_panel(rows, folds, traces):
    records = _valid_rows(rows)
    fold_map = {f["id"]: f for f in folds}
    if len(fold_map) != len(folds):
        raise ValueError("duplicate fold")
    runs = set()
    for fold in folds:
        train, evaluation = set(fold["train_response_ids"]), set(fold["evaluation_response_ids"])
        if not train or not evaluation or train & evaluation or not train | evaluation <= set(records):
            raise ValueError("invalid fold populations")
        train_groups = {records[i]["group_id"] for i in train}
        eval_groups = {records[i]["group_id"] for i in evaluation}
        if train_groups & eval_groups:
            raise ValueError("training/evaluation source groups overlap")
        if {records[i]["benchmark_id"] for i in train} & {records[i]["benchmark_id"] for i in evaluation}:
            raise ValueError("whole-benchmark exclusion violated")
        # Selected source groups must preserve every configuration and trial.
        for chosen, groups in ((train, train_groups), (evaluation, eval_groups)):
            if chosen != {i for i, r in records.items() if r["group_id"] in groups}:
                raise ValueError("incomplete source-group membership")
    for trace in traces:
        if trace["run_id"] in runs or trace["fold_id"] not in fold_map:
            raise ValueError("unknown fold or duplicate run")
        runs.add(trace["run_id"])
        fold = fold_map[trace["fold_id"]]
        evaluation = set(fold["evaluation_response_ids"])
        for pair, selection in trace["pairs"].items():
            support, targets = selection["support"], selection["targets"]
            if len(support) != 31 or len(set(support)) != 31 or not targets or len(set(targets)) != len(targets):
                raise ValueError("complete 31-label history and unique target trials required")
            if not set(support + targets) <= evaluation or set(support) & set(targets):
                raise ValueError("support/target lineage violated")
            if any(records[i]["pair_id"] != pair for i in support + targets):
                raise ValueError("cross-pair contamination")
            if len({records[i]["item_key"] for i in support}) != 31:
                raise ValueError("support must preserve31 distinct purchased items")
            if {records[i]["group_id"] for i in support} & {records[i]["group_id"] for i in targets}:
                raise ValueError("source group appears in support and target")
        if "target_groups" in trace:
            expected = {i for i in evaluation if records[i]["group_id"] in trace["target_groups"]}
            actual = {i for p in trace["pairs"].values() for i in p["targets"]}
            if actual != expected:
                raise ValueError("complete target groups/configurations/trials required")
    return records


def _lineage(rows, folds):
    """Visible configuration audit; no equality inferred from public raw IDs."""
    subjects = {r["subject_key"]: r["subject"] for r in rows}
    names = defaultdict(set)
    for key, subject in subjects.items():
        names[subject.get("normalized_name", "").strip().casefold()].add(key)
    records = {r["response_id"]: r for r in rows}
    results = []
    for fold in folds:
        train = {records[i]["subject_key"] for i in fold["train_response_ids"]}
        evaluation = {records[i]["subject_key"] for i in fold["evaluation_response_ids"]}
        train_names = {subjects[k].get("normalized_name", "").strip().casefold() for k in train}
        evaluation_names = {subjects[k].get("normalized_name", "").strip().casefold() for k in evaluation}
        results.append(
            {
                "fold": fold["id"],
                "training_configurations": len(train),
                "evaluation_configurations": len(evaluation),
                "exact_configuration_overlap": len(train & evaluation),
                "visible_model_name_overlap": sorted((train_names & evaluation_names) - {""}),
                "missing_model_names": sum(
                    not subjects[k].get("normalized_name", "").strip() for k in train | evaluation
                ),
            }
        )
    return {
        "schema": "structured-visible-lineage/v1",
        "subject_configuration_count": len(subjects),
        "configuration_hashes": {k: digest(v) for k, v in subjects.items()},
        "folds": results,
        "claim": "benchmark transfer; subject/base-model independence is not asserted",
        "unresolved_base_model_lineage": "visible names/configurations are audited; absent canonical base lineage remains unknown",
    }


def _prepare_matched(source, cache, output):
    from research.proxy_gpu.protocol import load_data, verify_source_completion

    verify_source_completion(source)
    manifest = read_json(cache / "experiment.json")
    completion = read_json(cache / "completion.json")
    if completion.get("status") != "COMPLETE":
        raise ValueError("GPU feature cache is not completed")
    from research.proxy_iteration.artifacts import checked_asset

    paths = {k: checked_asset(cache, v) for k, v in manifest["assets"].items()}
    rows, folds, traces, _ = load_data(paths)
    # Bind every reused feature to its immutable completion, not its filename.
    for name in ("minilm", "lexical_head_tail_wordpiece", "measurementdb", "subjects"):
        relative = f"features/{name}.npy"
        binding = {"path": relative, **completion["outputs"][relative]}
        copy_once(checked_asset(cache, binding), output, f"features/{name}.npy")
    copy_once(paths["historical_lexical_head_tail"], output, "features/pool.npy")
    keys = read_json(paths["item-keys.json"])
    subject_keys = sorted({r["subject_key"] for r in rows})
    write_once(output / "features/keys.json", {"item_keys": keys, "subject_keys": subject_keys})
    _write_rows(output / "inputs/rows.jsonl", rows)
    write_once(output / "inputs/folds.json", folds)
    write_once(output / "inputs/traces.json", traces)
    copy_once(cache / "features/receipt.json", output, "inputs/feature-source-receipt.json")
    copy_once(cache / "experiment.json", output, "inputs/cache-experiment.json")
    copy_once(cache / "completion.json", output, "inputs/cache-completion.json")
    copy_once(source / "experiment.json", output, "inputs/source-experiment.json")
    # Exact archived lexical forecasts are still a regression reference.
    copy_once(paths["archival_lexical"], output, "inputs/archival-lexical.jsonl")
    return (
        rows,
        folds,
        traces,
        {
            "source_manifest": file_identity(source / "experiment.json"),
            "gpu_cache_manifest": file_identity(cache / "experiment.json"),
        },
    )


def _prepare_broader(source, output):
    from research.recipe_eval.training import training_preflight
    from research.roadmap.replay import load_panel

    admission = read_json(source / "inputs/ADMISSION.json")
    panel = load_panel(
        (source / "inputs/source.jsonl").read_bytes(),
        expected_sha256=admission["source_sha256"],
        admission_sha256=file_identity(source / "inputs/ADMISSION.json")["sha256"],
        role="development",
        source_groups=(source / "inputs/groups.json").read_bytes(),
        source_groups_sha256=admission["groups_sha256"],
    )
    original = panel.record_map()
    rows = []
    for row in original.values():
        subject, item = row["input"]
        rows.append(
            {
                **{k: row[k] for k in ("response_id", "benchmark_id", "pair_id", "outcome", "item_id")},
                "subject_key": row["subject_id"],
                "item_key": "item:" + digest(item),
                "group_id": row["source_group_id"],
                "subject": subject,
                "item": item,
                "source_occurrence_index": row["source_occurrence_index"],
                "family_id": row["family_id"],
            }
        )
    records = {r["response_id"]: r for r in rows}
    folds, traces, legacy = [], [], {}
    used_ids = set()
    for cohort in ("A", "B"):
        ids = read_json(source / f"inputs/training-ids-{cohort}.json")
        legacy[cohort] = training_preflight({i: original[i] for i in ids})
        if not legacy[cohort]["passed"]:
            raise ValueError("legacy broader training admission changed")
        used_ids.update(ids)
        for family in ("bfcl", "livecodebench", "swebench"):
            label = family + "-" + cohort
            evaluation = read_json(source / f"inputs/evaluation-ids-{label}.json")
            # Validate every admission-bound ID file before admitting labels.
            for path, expected in (
                (source / f"inputs/evaluation-ids-{label}.json", admission["evaluation_ids"][label]["sha256"]),
                (
                    source / f"inputs/training-ids-{cohort}.json",
                    admission["fit_cohorts"][cohort]["training_ids"]["sha256"],
                ),
            ):
                if file_identity(path)["sha256"] != expected:
                    raise ValueError("broader admission file changed")
            raw = read_json(source / f"prediction-{label}/traces/0.json")
            frozen = read_json(source / f"prediction-{label}/frozen.json")
            if frozen["traces"] != [
                {
                    "path": "traces/0.json",
                    "sha256": file_identity(source / f"prediction-{label}/traces/0.json")["sha256"],
                }
            ]:
                raise ValueError("saved trace differs from its original frozen declaration")
            if (
                raw["source_sha256"] != admission["source_sha256"]
                or raw["admission_sha256"] != file_identity(source / "inputs/ADMISSION.json")["sha256"]
            ):
                raise ValueError("broader trace provenance differs")
            pairs = defaultdict(lambda: {"support": [], "targets": []})
            for rid in raw["target_response_ids"]:
                pairs[records[rid]["pair_id"]]["targets"].append(rid)
            for rid in raw["supports_by_budget"]["31"]:
                pairs[records[rid]["pair_id"]]["support"].append(rid)
            for budget in BUDGETS:
                expected = {i for p in pairs.values() for i in p["support"][:budget]}
                if expected != set(raw["supports_by_budget"][str(budget)]):
                    raise ValueError("broader prefixes do not match the original saved history")
            folds.append(
                {
                    "id": label,
                    "train_response_ids": ids,
                    "evaluation_response_ids": evaluation,
                    "training_cohort": cohort,
                    "axis": "benchmark",
                }
            )
            traces.append(
                {
                    "run_id": "broader-" + label + "-" + raw["run_id"],
                    "fold_id": label,
                    "pairs": dict(pairs),
                    "original_trace_sha256": file_identity(source / f"prediction-{label}/traces/0.json")["sha256"],
                }
            )
            used_ids.update(evaluation)
            copy_once(source / f"prediction-{label}/traces/0.json", output, f"inputs/original-traces/{label}.json")
            copy_once(source / f"prediction-{label}/frozen.json", output, f"inputs/original-traces/{label}-frozen.json")
    rows = sorted((r for r in rows if r["response_id"] in used_ids), key=lambda r: r["response_id"])
    _write_rows(output / "inputs/rows.jsonl", rows)
    write_once(output / "inputs/folds.json", folds)
    write_once(output / "inputs/traces.json", traces)
    write_once(output / "inputs/legacy-training-admission.json", legacy)
    for name in (
        "ADMISSION.json",
        "groups.json",
        "SCHEMA-PREFLIGHT.json",
        "TRAINING-PLAN-A.json",
        "TRAINING-PLAN-B.json",
    ):
        copy_once(source / "inputs" / name, output, "inputs/original-" + name)
    return (
        rows,
        folds,
        traces,
        {
            "source": file_identity(source / "inputs/source.jsonl"),
            "admission": file_identity(source / "inputs/ADMISSION.json"),
            "measurementdb": "not silently joined: broader common-coverage availability must be separately established",
        },
    )


def prepare(*, matched, gpu_cache, broader, root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    result = {}
    for track in ("matched", "broader"):
        output = root / track
        output.mkdir()
        if track == "matched":
            rows, folds, traces, provenance = _prepare_matched(Path(matched), Path(gpu_cache), output)
        else:
            rows, folds, traces, provenance = _prepare_broader(Path(broader), output)
        validate_panel(rows, folds, traces)
        write_once(output / "inputs/lineage-audit.json", _lineage(rows, folds))
        result[track] = {
            "trials": len(rows),
            "pairs": len({r["pair_id"] for r in rows}),
            "items": len({r["item_key"] for r in rows}),
            "folds": len(folds),
            "histories": len(traces),
            "benchmarks": dict(Counter(r["benchmark_id"] for r in rows)),
            "provenance": provenance,
        }
    write_once(
        root / "DATA-ADMISSION.json",
        {
            "schema": "structured-data-admission/v1",
            "tracks": result,
            "role": "exposed_development",
            "untouched_confirmation": False,
        },
    )
    from .legacy import collect_legacy

    forecasts, legacy = collect_legacy(root, broader)
    _write_rows(root / "broader/inputs/legacy-forecasts.jsonl", forecasts)
    write_once(root / "broader/inputs/legacy-reuse.json", legacy)
    seal_admission(root, gpu_cache=gpu_cache, broader=broader)
    return result


def load_panel_data(root, track):
    base = Path(root) / track
    rows = jsonl(base / "inputs/rows.jsonl")
    folds, traces = read_json(base / "inputs/folds.json"), read_json(base / "inputs/traces.json")
    records = validate_panel(rows, folds, traces)
    return rows, folds, traces, records


def seal_admission(root, *, gpu_cache, broader):
    """Close the materialization chain against original frozen source receipts."""
    from research.proxy_iteration.artifacts import checked_asset

    root, gpu_cache, broader = Path(root), Path(gpu_cache), Path(broader)
    cache = read_json(gpu_cache / "experiment.json")
    rows, folds, traces, _ = load_panel_data(root, "matched")
    if rows != jsonl(checked_asset(gpu_cache, cache["assets"]["cohort.jsonl"])):
        raise ValueError("matched response population differs from frozen cache")
    if folds != read_json(checked_asset(gpu_cache, cache["assets"]["folds.json"])) or traces != read_json(
        checked_asset(gpu_cache, cache["assets"]["traces"])
    ):
        raise ValueError("matched fold/history materialization differs")
    _, folds, traces, records = load_panel_data(root, "broader")
    trace_map = {t["fold_id"]: t for t in traces}
    for fold in folds:
        label = fold["id"]
        source = broader / ("prediction-" + label)
        frozen = read_json(source / "frozen.json")
        controller = read_json(source / "outer-controller.json")
        if (
            controller.get("status") != "complete"
            or controller.get("report_sha256") != file_identity(source / "report.json")["sha256"]
        ):
            raise ValueError("original broader controller/report chain is not complete")
        raw = read_json(source / "traces/0.json")
        if frozen["traces"] != [{"path": "traces/0.json", "sha256": file_identity(source / "traces/0.json")["sha256"]}]:
            raise ValueError("broader trace differs from original frozen declaration")
        if read_json(root / "broader/inputs/original-traces" / (label + ".json")) != raw:
            raise ValueError("copied broader history differs")
        trace = trace_map[label]
        expected_targets = {i for p in trace["pairs"].values() for i in p["targets"]}
        if expected_targets != set(raw["target_response_ids"]):
            raise ValueError("broader target trials/configurations changed")
        for budget in BUDGETS:
            actual = {i for p in trace["pairs"].values() for i in p["support"][:budget]}
            if actual != set(raw["supports_by_budget"][str(budget)]):
                raise ValueError("broader purchased history changed")
        actual_pairs = set(trace["pairs"])
        if actual_pairs != {records[i]["pair_id"] for i in raw["target_response_ids"]}:
            raise ValueError("broader evaluation pair population changed")
        if fold["train_response_ids"] != read_json(broader / f"inputs/training-ids-{fold['training_cohort']}.json"):
            raise ValueError("broader actual fit population changed")
        if not (root / "broader/inputs/original-traces" / (label + "-frozen.json")).exists():
            copy_once(source / "frozen.json", root, f"broader/inputs/original-traces/{label}-frozen.json")
    assets = {}
    for track in ("matched", "broader"):
        for path in sorted((root / track / "inputs").rglob("*")):
            if path.is_file():
                relative = path.relative_to(root).as_posix()
                assets[relative] = {"path": relative, **file_identity(path)}
    receipt = {
        "schema": "structured-admission-verification/v1",
        "status": "PASS",
        "assets": assets,
        "checks": [
            "exact_matched_population",
            "original_frozen_histories",
            "complete_broader_controller_reports",
            "all_budgets_all_pairs_all_target_trials",
            "complete_source_groups",
            "actual_fit_population",
            "full_visible_configuration_consistency",
        ],
        "confirmation": "EXPOSED_DEVELOPMENT_ONLY",
        "base_model_independence": "NOT_ESTABLISHED",
    }
    write_once(root / "ADMISSION-VERIFICATION.json", receipt)
    return receipt


def training_histories(rows, train_ids, saved=None):
    """Three input/source-group-derived training-only calibration histories."""
    records = {r["response_id"]: r for r in rows if r["response_id"] in set(train_ids)}
    if saved is not None:
        applicable = [
            t for t in saved if all(i in records for p in t["pairs"].values() for i in p["support"] + p["targets"])
        ]
        if len(applicable) != 3:
            raise ValueError("three exact original training-only histories required")
        return applicable
    by_pair = defaultdict(list)
    for row in records.values():
        by_pair[row["pair_id"]].append(row)
    traces = []
    for seed in HISTORIES:
        pairs = {}
        for pair, values in sorted(by_pair.items()):
            groups = sorted(
                {r["group_id"] for r in values}, key=lambda g: hashlib.sha256(canonical([seed, g])).hexdigest()
            )
            if len(groups) < 32:
                raise ValueError("complete six-budget calibration needs at least32 source groups per pair")
            support_groups = set(groups[:31])
            support = [
                min((r for r in values if r["group_id"] == group), key=lambda r: digest([seed, r["response_id"]]))[
                    "response_id"
                ]
                for group in groups[:31]
            ]
            targets = [r["response_id"] for r in values if r["group_id"] not in support_groups]
            pairs[pair] = {"support": support, "targets": sorted(targets)}
        traces.append({"run_id": "training-history-" + str(seed), "pairs": pairs})
    return traces
