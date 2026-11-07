"""Signed feature hashing (stable across processes; numpy + stdlib only)."""
import re
import zlib

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-_][a-z0-9]+)*")


def tokenize(text: str, max_chars: int = 4000) -> list:
    return _TOKEN.findall(text[:max_chars].lower())


def hash_tokens(vec: np.ndarray, tokens, prefix: str = "", weight: float = 1.0) -> None:
    d = len(vec)
    for t in tokens:
        h = zlib.crc32((prefix + t).encode())
        vec[h % d] += weight if (h >> 16) & 1 else -weight


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec


def parse_kv(s: str, sep: str = ";") -> dict:
    out = {}
    for part in s.split(sep):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out
