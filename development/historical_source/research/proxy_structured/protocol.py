"""Frozen matrix, numerical release gates and create-once source identities."""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
import re
from pathlib import Path

from research.proxy_iteration.artifacts import checked_asset, file_identity, read_json, write_once

SEEDS = (17, 29, 1013)
BUDGETS = (0, 1, 3, 7, 15, 31)
ALC_WEIGHTS = (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)
STRUCTURED = (
    ("rasch", "common"),
    ("twopl", "common"),
    ("threepl_global", "common"),
    ("threepl_item", "common"),
    ("mirt", "common"),
    ("caimira", "common"),
    ("rasch", "metadata"),
    ("caimira", "metadata"),
)
REPRESENTATIONS = {
    "matched": ("minilm", "lexical_head_tail_wordpiece", "measurementdb"),
    "broader": ("minilm", "lexical_head_tail_wordpiece"),
}
BLOCK_LIMITS = {
    "admission": 3600,
    "controls": 5400,
    "scalar_tabular": 5400,
    "multidimensional": 5400,
    "adaptation_replication": 3600,
    "audit": 5400,
}
PHASE_COMPLETIONS = {
    "core": "core-completion.json",
    "replication": "replication-completion.json",
    "tabular920": "tabular-9.2.0-completion.json",
    "tabular221": "tabular-2.2.1-completion.json",
}
PREFLIGHT_PROOF = re.compile(r"preflight-([A-Za-z0-9_-]{1,64})\.json")


def structured_name(representation, kind, prior):
    return representation + "__" + kind + ("_metadata" if prior == "metadata" else "")


def candidate_names(track):
    names = ["pm8", "fixed_pool", "ess_pool"]
    if track == "matched":
        names.append("archival_lexical")
    else:
        names.append("legacy_irt")
    fitted = [
        "lexical",
        "minilm_logistic",
        "minilm_ridge01",
        "minilm_uniform",
        "boost_shallow",
        "boost_regularized",
        "pool_calibrated",
    ]
    for representation in REPRESENTATIONS[track]:
        for kind, prior in STRUCTURED:
            base = structured_name(representation, kind, prior)
            fitted.extend((base + "_none", base + "_map"))
            if representation == "minilm" and kind in ("rasch", "twopl", "threepl_global", "threepl_item"):
                fitted.append(base + "_posterior")
    for version in ("221", "920"):
        fitted.extend(("tabpfn" + version + "_none", "tabpfn" + version + "_map"))
    names.extend(f"{name}.s{seed}" for name in fitted for seed in SEEDS)
    return names


def contrasts(track):
    result = [
        ("ess_pool", "fixed_pool"),
        ("pool_calibrated", "fixed_pool"),
        ("boost_shallow", "minilm_logistic"),
        ("boost_regularized", "minilm_logistic"),
        ("minilm_ridge01", "minilm_logistic"),
        ("tabpfn920_none", "minilm_uniform"),
        ("tabpfn920_none", "tabpfn221_none"),
    ]
    for representation in REPRESENTATIONS[track]:

        def name(kind, prior="common", representation=representation):
            return structured_name(representation, kind, prior)

        result += [
            (name("twopl") + "_map", name("rasch") + "_map"),
            (name("threepl_global") + "_map", name("twopl") + "_map"),
            (name("threepl_item") + "_map", name("threepl_global") + "_map"),
            (name("mirt") + "_map", name("twopl") + "_map"),
            (name("caimira") + "_map", name("mirt") + "_map"),
            (name("rasch", "metadata") + "_map", name("rasch") + "_map"),
            (name("caimira", "metadata") + "_map", name("caimira") + "_map"),
        ]
    return [list(pair) for pair in result]


def implementation_identity():
    repo = Path(__file__).resolve().parents[2]
    from research.proxy_gpu.protocol import implementation_identity as previous

    result = previous()
    for path in sorted((repo / "research/proxy_structured").glob("*.py")):
        result[path.relative_to(repo).as_posix()] = file_identity(path)
    # Runner imports beyond the inherited identity; src/torch_measure is never imported.
    for name in (
        "research/recipe_eval/__init__.py",
        "research/recipe_eval/subprocess_env.py",
        "research/recipe_eval/training.py",
        "research/recipe_eval/validation.py",
        "research/roadmap/candidates.py",
        "research/roadmap/replay.py",
        "research/roadmap/source_groups.py",
        "pyproject.toml",
        "docs/plans/gpu-structured-irt.md",
        "research/proxy_structured/requirements-tabpfn920.txt",
        "research/proxy_structured/requirements-tabpfn221.txt",
    ):
        result[name] = file_identity(repo / name)
    return result


def runtime_versions():
    """Resolved distribution versions and installation-record identities.

    Overlay environments disclose shadowed versions rather than pretending
    they are clean package-only contrasts. No installation path is persisted.
    """
    visible = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            visible.setdefault(name.lower().replace("_", "-"), set()).add(distribution.version)
    result = {}
    for name, versions in sorted(visible.items()):
        distribution = importlib.metadata.distribution(name)
        result[name] = {
            "resolved_version": distribution.version,
            "visible_versions": sorted(versions),
            "metadata_sha256": hashlib.sha256((distribution.read_text("METADATA") or "").encode()).hexdigest(),
            "record_sha256": hashlib.sha256((distribution.read_text("RECORD") or "").encode()).hexdigest(),
        }
    return result


def input_assets(root):
    root = Path(root)
    result = {}
    for track in ("matched", "broader"):
        for directory in (root / track / "inputs", root / track / "features"):
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    relative = path.relative_to(root).as_posix()
                    result[relative] = {"path": relative, **file_identity(path)}
    result["DATA-ADMISSION.json"] = {"path": "DATA-ADMISSION.json", **file_identity(root / "DATA-ADMISSION.json")}
    admission = read_json(root / "ADMISSION-VERIFICATION.json")
    if admission.get("status") != "PASS" or admission.get("schema") != "structured-admission-verification/v1":
        raise ValueError("materialization closure audit required")
    for binding in admission["assets"].values():
        checked_asset(root, binding)
    result["ADMISSION-VERIFICATION.json"] = {
        "path": "ADMISSION-VERIFICATION.json",
        **file_identity(root / "ADMISSION-VERIFICATION.json"),
    }
    return result


def _bounded(value, limit):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= limit


def validate_preflight(proof):
    """A free-form PASS cannot replace measured complete precision contracts."""
    from research.proxy_gpu.runtime import EXPECTED_VERSIONS

    if (
        proof.get("schema") != "proxy-structured-numerical-preflight/v1"
        or proof.get("status") != "PASS"
        or proof.get("cuda_qualified") is not True
    ):
        raise ValueError("actual complete CUDA qualification required")
    if proof.get("source_identity") != implementation_identity():
        raise ValueError("qualification source differs")
    if not proof.get("runtime_versions"):
        raise ValueError("complete runtime version inventory required")
    runtime = proof.get("runtime", {})
    if (
        any(runtime.get(k) != v for k, v in EXPECTED_VERSIONS.items())
        or runtime.get("cuda_available") is not True
        or runtime.get("pin_mismatches") != {}
    ):
        raise ValueError("qualified CUDA runtime pins differ")
    for category in ("models", "inference"):
        result = proof.get(category, {})
        if result.get("status") != "PASS":
            raise ValueError("missing model/inference qualification")
        for dtype, tolerance in (("float64", 1e-8), ("float32", 1e-5)):
            precision = result.get(dtype, {})
            difference = precision.get("max_probability_difference")
            if (
                type(difference) not in (int, float)
                or not math.isfinite(difference)
                or not 0 <= difference <= tolerance
            ):
                raise ValueError("invalid measured precision difference")
            if (
                precision.get("status") != "PASS"
                or not precision.get("cases")
                or any(c.get("status") != "PASS" for c in precision["cases"])
            ):
                raise ValueError("incomplete precision cases")
            if dtype == "float64" and (
                precision.get("objectives_pass") is not True or precision.get("gradients_pass") is not True
            ):
                raise ValueError("float64 objective/gradient preservation failed")
            if category == "models":
                declared = {(c.get("kind"), c.get("prior_kind")) for c in precision["cases"]}
                if declared != set(STRUCTURED) or len(precision["cases"]) != len(STRUCTURED):
                    raise ValueError("every structured recipe must qualify")
                for case in precision["cases"]:
                    if any(
                        case.get(key) is not True
                        for key in (
                            "finite_pass",
                            "initialization_hashes_equal",
                            "convergence_pass",
                            "failure_dispositions_pass",
                        )
                    ):
                        raise ValueError("recipe preservation disposition failed")
                    if case.get("trajectory_gate_applied") != (dtype == "float64"):
                        raise ValueError("precision trajectory policy differs")
            else:
                for case in precision["cases"]:
                    if (
                        any(case.get(key) is not True for key in ("finite_pass", "failure_dispositions_pass"))
                        or case.get("cpu_disposition") != case.get("device_disposition")
                        or not _bounded(case.get("max_probability_difference"), tolerance)
                        or not _bounded(case.get("finite_difference_error"), 1e-7)
                        or case.get("trajectory_gate_applied") != (dtype == "float64")
                        or dtype == "float64"
                        and any(
                            case.get(key) is not True for key in ("objectives_pass", "gradients_pass", "hessians_pass")
                        )
                    ):
                        raise ValueError("inference preservation disposition failed")
    posterior = proof["inference"].get("posterior", {})
    if (
        posterior.get("status") != "PASS"
        or not posterior.get("cases")
        or any(c.get("status") != "PASS" for c in posterior["cases"])
    ):
        raise ValueError("independent posterior references required")
    return proof


def runtime_proof(root, tabpfn920_proof, tabpfn221_proof, output):
    root = Path(root).resolve()
    proofs, statuses = {}, []
    for version, supplied in (("9.2.0", tabpfn920_proof), ("2.2.1", tabpfn221_proof)):
        path = Path(supplied).resolve()
        if not path.is_relative_to(root):
            raise ValueError("runtime proof must be inside the caller-owned evidence root")
        value = read_json(path)
        if (
            value.get("source_identity") != implementation_identity()
            or value.get("tabpfn", {}).get("package_version") != version
        ):
            raise ValueError("exact-source version-specific optional proof required")
        proofs[version] = {"path": path.relative_to(root).as_posix(), **file_identity(path)}
        statuses.append(value.get("status"))
    if any(s not in {"PASS", "FAIL"} for s in statuses):
        raise ValueError("completed measured optional qualification required")
    receipt = {
        "schema": "proxy-structured-optional-runtimes/v1",
        "proofs": proofs,
        "status": "PASS" if all(s == "PASS" for s in statuses) else "PASS_WITH_BLOCKED_RECIPES",
        "policy": "failed optional numerical contracts block their declared recipes; no silent CPU fallback",
        "comparison_scope": "installed procedures; unavoidable legacy Transformers/Hub differences are retained",
        "historical_gate": "PR7 accuracy failure preserved; no migration quality promotion",
    }
    write_once(output, receipt)
    return receipt


def _guarded_preflight(root, supplied):
    """Return a root-contained preflight proof and the completed guard receipt that produced it."""
    root = Path(root).resolve()
    path = Path(supplied).resolve()
    match = PREFLIGHT_PROOF.fullmatch(path.name)
    if path.parent != root or match is None:
        raise ValueError("preflight proof must be a preflight-<attempt>.json inside the evidence root")
    attempt = match.group(1)
    receipt_path = root / ("preflight-" + attempt + "-exit.json")
    if receipt_path.is_symlink() or receipt_path.resolve().parent != root:
        raise ValueError("preflight proof requires a guard receipt that is an ordinary file in the evidence root")
    receipt = read_json(receipt_path) if receipt_path.is_file() else {}
    expected_command = ["-m", "research.proxy_structured", "_worker", "--root", str(root)]
    expected_command += ["--mode", "preflight", "--attempt", attempt]
    if (
        receipt.get("schema") != "proxy-structured-external-worker/v1"
        or receipt.get("state") != "completed"
        or receipt.get("success") is not True
        or receipt.get("exit_code") != 0
        or (receipt.get("command") or [None])[1:] != expected_command
    ):
        raise ValueError("preflight proof requires its completed guarded worker receipt")
    return path, receipt_path


def freeze(root, preflight, runtime_receipt):
    root = Path(root)
    proof_path, receipt_path = _guarded_preflight(root, preflight)
    proof = read_json(proof_path)
    validate_preflight(proof)
    environment = read_json(runtime_receipt)
    if environment.get("schema") != "proxy-structured-optional-runtimes/v1" or environment.get("status") not in {
        "PASS",
        "PASS_WITH_BLOCKED_RECIPES",
    }:
        raise ValueError("isolated runtime qualification required")
    proofs = {version: read_json(checked_asset(root, environment["proofs"][version])) for version in ("9.2.0", "2.2.1")}
    optional = {}
    for version, optional_proof in proofs.items():
        if (
            optional_proof.get("source_identity") != implementation_identity()
            or optional_proof.get("runtime_identity", {}).get("torch") != "2.10.0+cu128"
        ):
            raise ValueError("optional qualification source/runtime identity differs")
        tabpfn = optional_proof.get("tabpfn", {})
        if (
            tabpfn.get("package_version") != version
            or tabpfn.get("checkpoint", {}).get("sha256")
            != "cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49"
        ):
            raise ValueError("optional checkpoint/version differs")
        difference = tabpfn.get("max_probability_difference")
        if type(difference) not in (int, float) or not math.isfinite(difference) or difference < 0:
            raise ValueError("optional precision measurement missing")
        passed = (
            difference <= 1e-5
            and tabpfn.get("duplicate_query_exact") is True
            and tabpfn.get("canonical_draw_hashes_match") is True
            and tabpfn.get("classes") == {"cpu": [0, 1], "cuda": [0, 1]}
            and tabpfn.get("one_class_dispositions")
            == {"cpu": "UNSUPPORTED_NO_FALLBACK", "cuda": "UNSUPPORTED_NO_FALLBACK"}
            and all(
                v.get("duplicate_exact") is True
                and v.get("order_max_probability_difference", 1) <= 1e-5
                and v.get("batch_split_max_probability_difference", 1) <= 1e-5
                for v in tabpfn.get("query_invariance", {}).values()
            )
        )
        if (tabpfn.get("status") == "PASS") != passed:
            raise ValueError("optional qualification falsely reports status")
        optional["tabpfn920" if version == "9.2.0" else "tabpfn221"] = passed
        if not optional_proof.get("runtime_versions"):
            raise ValueError("optional runtime installation inventory required")
    baseline = proofs["9.2.0"].get("baselines", {}).get("checks", {})
    for category in ("dense", "calibration", "pooling"):
        if not baseline.get(category) or any(c.get("status") != "PASS" for c in baseline[category]):
            raise ValueError("core baseline qualification failed: " + category)
    for recipe in ("shallow", "regularized"):
        cases = [c for c in baseline.get("boosted", []) if c.get("recipe") == recipe]
        if len(cases) != 1 or not math.isfinite(cases[0].get("max_probability_difference", float("nan"))):
            raise ValueError("boosting qualification incomplete")
        optional["boost_" + recipe] = cases[0]["status"] == "PASS" and cases[0]["max_probability_difference"] <= 1e-5
    blocked = sorted(
        name + suffix
        for name, passed in optional.items()
        if not passed
        for suffix in (("_none", "_map") if name.startswith("tabpfn") else ("",))
    )
    replication = {
        "schema": "proxy-structured-replication/v1",
        "source_identity": implementation_identity(),
        "assets": input_assets(root),
        "structured_recipes": [list(v) for v in STRUCTURED],
        "representations": {k: list(v[1:]) for k, v in REPRESENTATIONS.items()},
        "seeds": list(SEEDS),
        "budget_seconds": BLOCK_LIMITS["adaptation_replication"],
        "selection": "all declared recipes; frozen before primary results; no winner-conditioned combinations",
    }
    replication_binding = write_once(root / "replication-experiment.json", replication)
    manifest = {
        "schema": "proxy-structured-experiment/v1",
        "source_identity": implementation_identity(),
        "replication_manifest": {"path": "replication-experiment.json", **replication_binding},
        "assets": input_assets(root),
        "preflight": proof,
        "preflight_binding": {
            "proof": {"path": proof_path.name, **file_identity(proof_path)},
            "guard_receipt": {"path": receipt_path.name, **file_identity(receipt_path)},
        },
        "environment": environment,
        "optional_qualification": optional,
        "blocked_procedures": blocked,
        "seeds": list(SEEDS),
        "budgets": list(BUDGETS),
        "alc_weights": list(ALC_WEIGHTS),
        "structured_recipes": [list(v) for v in STRUCTURED],
        "representations": REPRESENTATIONS,
        "tracks": {
            track: {
                "candidates": candidate_names(track),
                "primary_contrasts": contrasts(track),
                "screen_references": ["pm8", "archival_lexical" if track == "matched" else "lexical"],
            }
            for track in ("matched", "broader")
        },
        "statistics": {"draws": 2000, "seed": 20261101, "minimum_valid": 1900},
        "gates": {
            "brier_gain": 0.002,
            "family_regression": 0.01,
            "early_regression": 0.005,
            "ece_regression": 0.005,
            "extra_fallback": 0,
        },
        "limits": {
            "total_seconds": 28800,
            "block_seconds": BLOCK_LIMITS,
            "fit_seconds": 1200,
            "host_rss_bytes": 32 * 1024**3,
            "gpu_reserved_bytes": 20 * 1024**3,
            "audit_reserve_seconds": 5400,
            "gpu_workers": 1,
        },
        "recipe": {
            "steps": 300,
            "optimizer": "Adam",
            "learning_rate": 0.01,
            "ridge": 0.01,
            "tf32": False,
            "autocast": False,
            "foreach": False,
            "fused": False,
            "precision": "float32 fits; float64 support inference; CPU canonical initial tensors",
            "rank": 2,
            "training_weights": "equal_pair; explicit uniform-context companions",
            "calibration": "single positive-slope Platt across all6 training budgets",
            "no_intermediate_winner_composites": True,
        },
        "confirmation_status": "EXPOSED_DEVELOPMENT_ONLY",
        "external_submissions": False,
        "historical_pr7_accuracy_gate": "FAIL_PRESERVED_1.0041279669762648pp_above1pp",
    }
    binding = write_once(root / "experiment.json", manifest)
    return {
        "manifest_sha256": binding["sha256"],
        "raw_candidates": {t: len(v["candidates"]) for t, v in manifest["tracks"].items()},
    }


def load(root, expected_hash=None, *, verify_assets=True):
    root = Path(root)
    if expected_hash is not None and file_identity(root / "experiment.json")["sha256"] != expected_hash:
        raise ValueError("manifest identity changed")
    manifest = read_json(root / "experiment.json")
    if (
        manifest.get("schema") != "proxy-structured-experiment/v1"
        or manifest["source_identity"] != implementation_identity()
    ):
        raise ValueError("source identity changed; preserve evidence and freeze successor")
    if verify_assets:
        for binding in manifest["assets"].values():
            checked_asset(root, binding)
        checked_asset(root, manifest["replication_manifest"])
        for binding in manifest["environment"]["proofs"].values():
            checked_asset(root, binding)
        for binding in manifest["preflight_binding"].values():
            checked_asset(root, binding)
    return manifest


def status(root, expected_hash):
    root = Path(root)
    manifest = load(root, expected_hash)
    jobs = sorted((root / "jobs").glob("*/receipt.json")) if (root / "jobs").exists() else []
    receipts = [read_json(path) for path in jobs]
    phases = {}
    for phase, completion in PHASE_COMPLETIONS.items():
        exit_path = root / (phase + "-exit.json")
        record = read_json(exit_path) if exit_path.is_file() else None
        phases[phase] = {
            "exit": None
            if record is None
            else {key: record.get(key) for key in ("state", "exit_code", "success", "elapsed_seconds")},
            "completion_present": (root / completion).is_file(),
        }
    result = {
        "schema": "proxy-structured-status/v1",
        "manifest_sha256": expected_hash,
        "declared_tracks": list(manifest["tracks"]),
        "jobs": len(receipts),
        "job_statuses": {s: sum(r["status"] == s for r in receipts) for s in sorted({r["status"] for r in receipts})},
        "phases": phases,
        "completion": read_json(root / "completion.json") if (root / "completion.json").exists() else None,
    }
    for receipt in receipts:
        for binding in receipt.get("outputs", {}).values():
            checked_asset(root, binding)
    return result
