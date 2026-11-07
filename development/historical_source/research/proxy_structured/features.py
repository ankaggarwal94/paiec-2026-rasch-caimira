"""Compute only absent visible-input features in a bounded CUDA worker."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from research.proxy_gpu.features import GPUTextEncoder, encode_subjects, lexical_scope_features
from research.proxy_gpu.runtime import check_resources, configure, resource_snapshot
from research.proxy_iteration.artifacts import file_identity, read_json, write_once
from research.roadmap.features import model_identity

from .data import load_panel_data


def prepare_features(root, encoder):
    root = Path(root)
    configure()
    started = time.monotonic()
    rows, _, _, _ = load_panel_data(root, "broader")
    directory = root / "broader/features"
    directory.mkdir(exist_ok=False)
    items, subjects = {}, {}
    for row in rows:
        items.setdefault(row["item_key"], row["item"])
        subjects.setdefault(row["subject_key"], row["subject"])
    keys, subject_keys = sorted(items), sorted(subjects)
    values = [items[key]["item_content"] for key in keys]
    # Every identical text is encoded once by GPUTextEncoder's token/chunk cache.
    model = GPUTextEncoder(encoder, model_identity(), batch_size=64)
    matrices = {
        "minilm": model.encode(values),
        "lexical_head_tail_wordpiece": lexical_scope_features(values, model.tokenizer)["lexical_head_tail_wordpiece"],
        "subjects": encode_subjects([subjects[key] for key in subject_keys], model),
    }
    from research.proxy_iteration.features import lexical_head_tail_features

    matrices["pool"], _ = lexical_head_tail_features(values, model.tokenizer)
    outputs = {}
    for name, matrix in matrices.items():
        if not np.isfinite(matrix).all():
            raise ArithmeticError("nonfinite visible feature")
        path = directory / (name + ".npy")
        with path.open("xb") as stream:
            np.save(stream, matrix, allow_pickle=False)
        outputs[path.name] = file_identity(path)
    write_once(directory / "keys.json", {"item_keys": keys, "subject_keys": subject_keys})
    check_resources()
    receipt = {
        "schema": "structured-feature-cache/v1",
        "status": "COMPLETE",
        "uses_outcome_labels": False,
        "inputs": file_identity(root / "broader/inputs/rows.jsonl"),
        "encoder": model_identity(),
        "recipe": "proxy_gpu.features.GPUTextEncoder head_tail float32 eager; lexical exact head-tail WordPiece",
        "batch_size": 64,
        "outputs": outputs,
        "metrics": model.metrics,
        "resource": resource_snapshot(),
        "elapsed_seconds": time.monotonic() - started,
        "measurementdb": "UNAVAILABLE_FULL_BROADER_COVERAGE; matched track retains complete published-vector control",
    }
    write_once(directory / "receipt.json", receipt)
    return receipt


def load_features(root, track):
    directory = Path(root) / track / "features"
    keys = read_json(directory / "keys.json")
    names = ("minilm", "lexical_head_tail_wordpiece", "measurementdb", "subjects", "pool")
    matrices = {
        name: np.load(directory / (name + ".npy"), allow_pickle=False)
        for name in names
        if (directory / (name + ".npy")).exists()
    }
    if not {"minilm", "lexical_head_tail_wordpiece", "subjects"} <= set(matrices):
        raise ValueError("complete common feature bank required")
    for name, matrix in matrices.items():
        count = len(keys["subject_keys"] if name == "subjects" else keys["item_keys"])
        if matrix.ndim != 2 or len(matrix) != count or not np.isfinite(matrix).all():
            raise ValueError("feature keys, rows or finite values differ")
    return matrices, {k: i for i, k in enumerate(keys["item_keys"])}, {k: i for i, k in enumerate(keys["subject_keys"])}
