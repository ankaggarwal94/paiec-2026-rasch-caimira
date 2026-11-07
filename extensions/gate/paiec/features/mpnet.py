"""Frozen all-mpnet-base-v2 item embeddings, precomputed once for the whole dataset (fit-time only).

Used by the CaimiraLite reference models (`predictors/ref_caimira.py`). Rows are keyed by a stable 64-bit hash of
`item_key`, so a predictor can look embeddings up from the item dicts it sees at prediction time.
"""
import hashlib
import os
from pathlib import Path

import numpy as np

CACHE = Path(os.environ.get("PAIEC_MPNET", "/fs/clip-scratch/mgor/paiec-cache/mpnet_items.npz"))
ENCODER = "sentence-transformers/all-mpnet-base-v2"
MAX_CHARS = 1000   # encoder max_seq_length is 128 tokens (their default 384): ~10x cheaper over 2.85M items


def item_text(item: dict) -> str:
    cond = item.get("interactors", "") or "none"
    return f"Benchmark: {item.get('benchmark_id', '')}\nCondition: {cond}\nItem:\n{item.get('item_content', '')[:MAX_CHARS]}"


def item_hash(item: dict) -> int:
    k = "\x1f".join((item.get("benchmark_id", ""), item.get("item_features", ""), item.get("interactors", ""),
                     item.get("item_content", "")))
    return int.from_bytes(hashlib.blake2b(k.encode(), digest_size=8).digest(), "little")


class EmbeddingTable:
    def __init__(self, path=CACHE):
        z = np.load(path)
        order = np.argsort(z["keys"])
        self.keys, self.emb = z["keys"][order], z["emb"][order]

    def lookup(self, items, dtype=np.float32) -> np.ndarray:
        h = np.array([item_hash(i) for i in items], dtype=np.uint64)
        pos = np.minimum(np.searchsorted(self.keys, h), len(self.keys) - 1)
        ok = self.keys[pos] == h
        out = self.emb[pos].astype(dtype, copy=False)
        out[~ok] = 0.0   # unseen item text: zero embedding (model falls back to biases)
        return out
