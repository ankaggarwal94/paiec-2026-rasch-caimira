"""Caller-owned roots and sequential externally bounded CUDA campaign actions."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PUBLIC_COMMANDS = (
    "prepare",
    "features",
    "preflight",
    "optional-preflight",
    "freeze",
    "runtime-proof",
    "run",
    "status",
    "audit",
    "qualify",
)


def _label(mode, version=None, attempt=None):
    """Log/receipt stem shared by a guarded controller and the worker it launches."""
    label = mode + (f"-{version}" if mode == "optional-preflight" else "")
    return label + (f"-{attempt}" if mode in {"preflight", "optional-preflight"} else "")


def _bind_review(root, review):
    """Persist the release review once; every later phase must present the same file."""
    from research.proxy_iteration.artifacts import copy_once, file_identity

    target = root / "release-review.json"
    if not target.exists():
        return copy_once(review, root, target.name)
    binding = {"path": target.name, **file_identity(target)}
    if file_identity(review) != {key: binding[key] for key in ("sha256", "bytes")}:
        raise ValueError("release review differs from the persisted review")
    return binding


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True, metavar="{" + ",".join(PUBLIC_COMMANDS) + "}")
    p = commands.add_parser("prepare")
    for name in ("matched", "gpu-cache", "broader", "root"):
        p.add_argument("--" + name, required=True)
    p = commands.add_parser("features")
    p.add_argument("--root", required=True)
    p.add_argument("--encoder", required=True)
    p = commands.add_parser("preflight")
    p.add_argument("--root", required=True)
    p.add_argument("--attempt", default="v1")
    p = commands.add_parser("optional-preflight")
    p.add_argument("--root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--version", choices=("2.2.1", "9.2.0"), required=True)
    p.add_argument("--attempt", default="v1")
    p = commands.add_parser("freeze")
    for name in ("root", "preflight", "runtime-receipt"):
        p.add_argument("--" + name, required=True)
    p = commands.add_parser("runtime-proof")
    for name in ("root", "tabpfn920-proof", "tabpfn221-proof", "output"):
        p.add_argument("--" + name, required=True)
    for action in ("run", "status", "audit", "qualify"):
        p = commands.add_parser(action)
        p.add_argument("--root", required=True)
        p.add_argument("--manifest-sha256", required=True)
        if action == "run":
            p.add_argument("--review", required=True)
            p.add_argument("--phase", choices=("core", "replication", "tabular920", "tabular221"), required=True)
            p.add_argument("--checkpoint")
    p = commands.add_parser("_worker")
    p.add_argument("--root", required=True)
    p.add_argument(
        "--mode",
        choices=(
            "features",
            "preflight",
            "optional-preflight",
            "core",
            "replication",
            "tabular920",
            "tabular221",
            "audit",
        ),
        required=True,
    )
    p.add_argument("--encoder")
    p.add_argument("--checkpoint")
    p.add_argument("--version")
    p.add_argument("--manifest-sha256")
    p.add_argument("--attempt", default="v1")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    if args.action == "prepare":
        from .data import prepare

        result = prepare(matched=args.matched, gpu_cache=args.gpu_cache, broader=args.broader, root=root)
    elif args.action == "freeze":
        from .protocol import freeze

        result = freeze(root, args.preflight, args.runtime_receipt)
    elif args.action == "runtime-proof":
        from .protocol import runtime_proof

        result = runtime_proof(root, args.tabpfn920_proof, args.tabpfn221_proof, args.output)
    elif args.action == "status":
        from .protocol import status

        result = status(root, args.manifest_sha256)
    elif args.action == "qualify":
        from .analysis import package_disposition

        result = package_disposition(root, args.manifest_sha256)
    elif args.action == "_worker":
        from research.proxy_iteration.artifacts import write_once

        from .guard import supervised_worker

        supervised_worker(root / (_label(args.mode, args.version, args.attempt) + "-exit.json"))
        if args.mode == "features":
            from .features import prepare_features

            result = prepare_features(root, args.encoder)
        elif args.mode == "preflight":
            from .protocol import implementation_identity, runtime_versions
            from .qualification import numerical_preflight

            before = implementation_identity()
            started = time.monotonic()
            result = numerical_preflight(device="cuda", include_models=True)
            if before != implementation_identity():
                raise ValueError("source changed during qualification")
            result.update(
                source_identity=before, runtime_versions=runtime_versions(), elapsed_seconds=time.monotonic() - started
            )
            write_once(root / ("preflight-" + args.attempt + ".json"), result)
        elif args.mode == "optional-preflight":
            from research.proxy_gpu.runtime import runtime_identity

            from .protocol import implementation_identity, runtime_versions
            from .tabular import qualify_tabpfn

            before = implementation_identity()
            result = {"tabpfn": qualify_tabpfn(args.checkpoint, expected_version=args.version)}
            if args.version == "9.2.0":
                from .baselines import qualify_baselines

                result["baselines"] = qualify_baselines()
            result["status"] = "PASS" if all(v.get("status") == "PASS" for v in result.values()) else "FAIL"
            if before != implementation_identity():
                raise ValueError("source changed during optional qualification")
            result.update(
                source_identity=before, runtime_identity=runtime_identity(), runtime_versions=runtime_versions()
            )
            write_once(root / ("optional-preflight-" + args.version + "-" + args.attempt + ".json"), result)
        elif args.mode == "audit":
            from .analysis import audit

            result = audit(root, args.manifest_sha256)
        elif args.mode in {"core", "replication"}:
            from .experiment import core_worker

            result = core_worker(root, args.manifest_sha256, replication=args.mode == "replication")
        else:
            from .experiment import tabular_worker

            result = tabular_worker(
                root, args.manifest_sha256, args.checkpoint, "9.2.0" if args.mode == "tabular920" else "2.2.1"
            )
    else:
        from research.proxy_gpu.runtime import device_lock_path
        from research.proxy_iteration.artifacts import file_identity, read_json, write_once

        from .guard import run_guarded

        command = [sys.executable, "-m", "research.proxy_structured", "_worker", "--root", str(root)]
        mode, maximum = args.action, 3600
        if args.action == "run":
            from .protocol import load

            load(root, args.manifest_sha256)
            review = read_json(args.review)
            if (
                review.get("manifest_sha256") != args.manifest_sha256
                or review.get("status") != "PASS"
                or not review.get("independent_reviewers")
            ):
                raise ValueError("independent manifest-bound release review required")
            mode, maximum = args.phase, 23400
            if args.phase in {"tabular920", "tabular221"}:
                from .tabular import checkpoint_identity

                if not args.checkpoint:
                    raise ValueError("explicit pinned checkpoint required")
                checkpoint_identity(args.checkpoint)
            release = _bind_review(root, args.review)
            if not (root / "scientific-start.json").exists():
                admission = sum(read_json(p).get("elapsed_seconds", 0) for p in root.glob("*-exit.json"))
                write_once(
                    root / "scientific-start.json",
                    {
                        "unix_seconds": time.time(),
                        "admission_seconds": admission,
                        "total_seconds": 28800,
                        "audit_reserve_seconds": 5400,
                        "prior_qualification_start_preserved": file_identity(root / "execution-start.json")
                        if (root / "execution-start.json").exists()
                        else None,
                        "release_review": release,
                        "semantics": "implementation precedes clock; actual qualification-worker seconds debit the60minute admission block and8hour aggregate",
                    },
                )
            elif read_json(root / "scientific-start.json").get("release_review") != release:
                raise ValueError("release review differs from the one bound at the scientific start")
            command.extend(("--manifest-sha256", args.manifest_sha256))
            if args.phase in {"tabular920", "tabular221"}:
                command.extend(("--checkpoint", args.checkpoint))
        elif args.action == "features":
            command.extend(("--encoder", args.encoder))
        elif args.action == "optional-preflight":
            command.extend(("--checkpoint", args.checkpoint, "--version", args.version))
        elif args.action == "audit":
            maximum = 5400
            command.extend(("--manifest-sha256", args.manifest_sha256))
        command.extend(("--mode", mode))
        if args.action in {"preflight", "optional-preflight"}:
            if not args.attempt or len(args.attempt) > 64 or any(not (c.isalnum() or c in "-_") for c in args.attempt):
                raise ValueError("bounded alphanumeric attempt label required")
            command.extend(("--attempt", args.attempt))
        if args.action in {"run", "audit"}:
            timing = read_json(root / "scientific-start.json")
            elapsed = time.time() - timing["unix_seconds"] + timing["admission_seconds"]
            remaining = 28800 - elapsed - (0 if args.action == "audit" else 5400)
        else:
            admission = sum(read_json(p).get("elapsed_seconds", 0) for p in root.glob("*-exit.json"))
            remaining = 3600 - admission
        maximum = min(maximum, remaining)
        if maximum <= 0:
            raise TimeoutError("eight-hour execution ceiling/audit reserve exhausted")
        label = _label(mode, getattr(args, "version", None), getattr(args, "attempt", None))
        if args.action == "run":
            (root / "stages" / mode).mkdir(parents=True, exist_ok=True)
        result = run_guarded(
            command,
            cwd=Path(__file__).resolve().parents[2],
            log_path=root / (label + ".log"),
            receipt_path=root / (label + "-exit.json"),
            max_seconds=maximum,
            stage_state_path=root / "stages" / mode / "current-stage.json" if args.action == "run" else None,
            lock_path=device_lock_path(),
        )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return int(isinstance(result, dict) and (result.get("success") is False or result.get("status") == "FAIL"))


if __name__ == "__main__":
    raise SystemExit(main())
