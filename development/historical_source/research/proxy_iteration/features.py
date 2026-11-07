"""Item-text features with explicit, shared WordPiece visibility controls.

MeasurementDB joining/admission belongs to the data pipeline. Validation here
does not manufacture missing vectors, infer provenance or authorize deployment.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from pathlib import PurePosixPath

import numpy as np

from research.five_candidate.models import MiniLMEncoder
from research.proxy_iteration.models import _contains_boolean

LEXICAL_FULL = "lexical_item_full_v1"
LEXICAL_HEAD_TAIL = "lexical_item_wordpiece_head_tail_254_v1"
MINILM_HEAD_TAIL = "minilm_item_head_tail_254_v1"
MEASUREMENTDB = "supplied_measurementdb1536_v1"
FEATURE_DIMENSIONS = {LEXICAL_FULL: 384, LEXICAL_HEAD_TAIL: 384, MINILM_HEAD_TAIL: 384, MEASUREMENTDB: 1536}


def validate_texts(texts):
    if not isinstance(texts, (list, tuple)) or any(not isinstance(t, str) for t in texts):
        raise ValueError("features require a sequence of exact original item strings")
    if any(len(t.encode("utf8")) > 64 * 1024**2 for t in texts):
        raise ValueError("item text exceeds the declared64-MiB bound")
    return texts


def _hash_tokens(tokens):
    counts = [0] * 384
    for token, count in Counter(tokens).items():
        h = hashlib.blake2b(token.encode("utf8"), digest_size=8, person=b"roadmap.lex.v1").digest()
        counts[int.from_bytes(h[:4], "little") % 384] += count if h[4] & 1 else -count
    norm = math.sqrt(math.fsum(x * x for x in counts))
    return [x / norm if norm else 0.0 for x in counts]


def lexical_features(texts):
    """Hash complete item text only; no subject, benchmark or JSON envelope."""
    validate_texts(texts)
    return np.asarray(
        [_hash_tokens(re.findall(r"\w+|[^\w\s]", t.casefold())) for t in texts], dtype=np.float32
    ).reshape(-1, 384)


def tokenize_texts(texts, tokenizer):
    """Return frozen first127/last127 WordPiece membership and row diagnostics."""
    validate_texts(texts)
    kept, metadata = [], []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False, truncation=False, verbose=False)
        if len(ids) > 8_000_000 or any(type(v) is not int or v < 0 for v in ids):
            raise ValueError("invalid or oversized tokenizer output")
        row = ids if len(ids) <= 254 else ids[:127] + ids[-127:]
        kept.append(row)
        metadata.append(
            {
                "content_sha256": hashlib.sha256(text.encode("utf8")).hexdigest(),
                "original_wordpieces": len(ids),
                "retained_wordpieces": len(row),
                "truncated": len(ids) > 254,
                "selection": "all_if_at_most254_else_first127_last127",
            }
        )
    return kept, metadata


def lexical_token_features(token_rows, tokenizer):
    """Hash the exact kept WordPiece strings, without decoding/re-tokenizing."""
    result = []
    for row in token_rows:
        if len(row) > 254 or any(type(v) is not int or v < 0 for v in row):
            raise ValueError("invalid kept WordPiece row")
        tokens = tokenizer.convert_ids_to_tokens(row)
        if len(tokens) != len(row) or any(not isinstance(t, str) for t in tokens):
            raise ValueError("tokenizer did not preserve retained-token membership")
        result.append(_hash_tokens(tokens))
    return np.asarray(result, dtype=np.float32).reshape(-1, 384)


def lexical_head_tail_features(texts, tokenizer):
    rows, metadata = tokenize_texts(texts, tokenizer)
    return lexical_token_features(rows, tokenizer), metadata


def measurementdb_features(vectors):
    """Validate already exactly joined1536-dimensional vectors; never renormalize."""
    if _contains_boolean(vectors):
        raise ValueError("MeasurementDB features must not contain Boolean values")
    values = np.asarray(vectors)
    if values.dtype.kind not in "fiu" or values.ndim != 2 or values.shape[1] != 1536:
        raise ValueError("MeasurementDB features must be a numeric N by1536 matrix")
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("MeasurementDB features must be finite")
    norms = np.sqrt(np.sum(values * values, axis=1))
    if np.any(np.abs(norms - 1.0) > 0.001):
        raise ValueError("MeasurementDB vectors violate the declared unit-norm tolerance")
    return values


class MiniLMTextEncoder(MiniLMEncoder):
    """Reuse pinned loading/math; encode plain original item text under a new ID."""

    def __init__(self, directory, manifest):
        for name in manifest.get("files", {}):
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
                raise ValueError("encoder manifest path must remain contained and relative")
        super().__init__(directory, manifest)

    def encode_token_ids(self, rows):
        if not 1 <= len(rows) <= 32:
            raise ValueError("encoder batches must contain1..32 rows")
        ids, masks = [], []
        for row in rows:
            if len(row) > 254 or any(type(v) is not int or v < 0 for v in row):
                raise ValueError("invalid retained WordPiece row")
            complete = [101, *row, 102]
            ids.append(complete + [0] * (256 - len(complete)))
            masks.append([1] * len(complete) + [0] * (256 - len(complete)))
        torch = self.torch
        with torch.inference_mode(), torch.autocast(device_type="cpu", enabled=False):
            tensor = torch.tensor(ids, dtype=torch.int64)
            mask = torch.tensor(masks, dtype=torch.int64)
            hidden = self.model(
                input_ids=tensor, attention_mask=mask, token_type_ids=torch.zeros_like(tensor), return_dict=True
            ).last_hidden_state
            expanded = mask.unsqueeze(-1).to(dtype=torch.float32)
            mean = (hidden.float() * expanded).sum(dim=1) / expanded.sum(dim=1).clamp(min=1e-9)
            result = torch.nn.functional.normalize(mean, p=2, dim=1).cpu().numpy()
        if result.shape != (len(rows), 384) or not np.isfinite(result).all():
            raise ArithmeticError("invalid MiniLM feature output")
        if np.any(np.abs(np.sqrt(np.sum(result.astype(np.float64) ** 2, axis=1)) - 1) > 1e-5):
            raise ArithmeticError("MiniLM feature output is not normalized")
        return result

    def encode(self, texts):
        rows, _ = tokenize_texts(texts, self.tokenizer)
        return self.encode_token_ids(rows)


def encode_matched_texts(texts, encoder, *, batch_size=16):
    """Reuse one tokenization for MiniLM and its lexical visibility control.

    Returns NumPy arrays plus metadata; no labels are accepted by this API.
    Caller freezes batch membership and binds complete input identities.
    """
    validate_texts(texts)
    if type(batch_size) is not int or not 1 <= batch_size <= 32:
        raise ValueError("batch size must be1..32")
    rows, metadata = tokenize_texts(texts, encoder.tokenizer)
    chunks = [encoder.encode_token_ids(rows[i : i + batch_size]) for i in range(0, len(rows), batch_size)]
    minilm = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 384), dtype=np.float32)
    return {
        "features": {
            LEXICAL_FULL: lexical_features(texts),
            LEXICAL_HEAD_TAIL: lexical_token_features(rows, encoder.tokenizer),
            MINILM_HEAD_TAIL: minilm,
        },
        "token_metadata": metadata,
        "batch_membership": [list(range(i, min(i + batch_size, len(rows)))) for i in range(0, len(rows), batch_size)],
        "feature_dimensions": {k: FEATURE_DIMENSIONS[k] for k in (LEXICAL_FULL, LEXICAL_HEAD_TAIL, MINILM_HEAD_TAIL)},
        "encoder_compatibility": encoder.compatibility,
    }
