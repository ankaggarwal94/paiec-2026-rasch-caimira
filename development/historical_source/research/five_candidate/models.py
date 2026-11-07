"""Portable predictors for the frozen five-family comparison.

The flat submission copies this exact module and the existing contract and
MAP numerical modules. No training data, response ID, or query lookup is bundled.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
from collections import Counter, OrderedDict
from pathlib import Path

if __package__:
    from profiles.neurips import contract
    from research.roadmap import numerics
else:
    import fc_contract as contract
    import fc_numerics as numerics

ARMS = ("pm4", "lexical_similarity", "lexical_direct", "minilm_direct")
DIMENSION = 384
ENCODER_ID = "sentence-transformers/all-MiniLM-L6-v2"
ENCODER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MAX_INPUT_BYTES = 64 * 1024**2


def canonical(value):
    contract.canonical_json(value)  # Reject lossy/non-finite Python values.
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def input_document(value):
    subject, item = contract.normalize_input(value)
    document = canonical([subject, item])
    if len(document.encode()) > MAX_INPUT_BYTES:
        raise ValueError("visible input exceeds 64 MiB")
    return document


def lexical384(value):
    """Exact roadmap lexical384_v1 transform, including float32 rounding."""
    counts = [0] * DIMENSION
    for token, count in Counter(re.findall(r"\w+|[^\w\s]", input_document(value).casefold())).items():
        h = hashlib.blake2b(token.encode(), digest_size=8, person=b"roadmap.lex.v1").digest()
        counts[int.from_bytes(h[:4], "little") % DIMENSION] += count if h[4] & 1 else -count
    norm = math.sqrt(math.fsum(x * x for x in counts))
    return tuple(struct.unpack("<f", struct.pack("<f", x / norm))[0] if norm else 0.0 for x in counts)


def validate_feature(feature):
    if len(feature) != DIMENSION:
        raise ValueError("feature must have 384 components")
    result = tuple(float(x) for x in feature)
    if not all(math.isfinite(x) for x in result):
        raise ValueError("non-finite feature")
    if any(struct.unpack("<f", struct.pack("<f", x))[0] != x for x in result):
        raise ValueError("features must retain exact float32 values")
    norm = math.sqrt(math.fsum(x * x for x in result))
    if norm and abs(norm - 1.0) > 1e-5:
        raise ValueError("feature must be zero or L2 normalized")
    return result


def _validate_checkpoint_loading(model, loading, checkpoint_path):
    """Admit only the verified, non-learned Transformers 5.5 buffer change."""
    unexpected = loading.get("unexpected_keys", [])
    if (
        not isinstance(unexpected, (list, tuple, set, frozenset))
        or len(unexpected) > 1
        or set(unexpected) not in (set(), {"embeddings.position_ids"})
    ) or any(value for key, value in loading.items() if key != "unexpected_keys"):
        raise ValueError("encoder checkpoint did not load completely")
    if not unexpected:
        return {"ignored_nonlearned_buffers": []}
    import torch
    from safetensors import safe_open

    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as stream:
        stored = stream.get_tensor("embeddings.position_ids")
    runtime = model.embeddings.position_ids.detach().cpu()
    expected = torch.arange(model.config.max_position_embeddings, dtype=torch.int64).expand(1, -1)
    if (
        stored.dtype != torch.int64
        or runtime.dtype != torch.int64
        or stored.shape != expected.shape
        or runtime.shape != expected.shape
        or not torch.equal(stored, expected)
        or not torch.equal(runtime, expected)
        or "position_ids" not in model.embeddings._non_persistent_buffers_set
    ):
        raise ValueError("unexpected position buffer is not the identical fixed runtime buffer")
    return {
        "ignored_nonlearned_buffers": ["embeddings.position_ids"],
        "position_ids_shape": list(expected.shape),
        "position_ids_dtype": "int64",
        "position_ids_equal_runtime_arange": True,
    }


class MiniLMEncoder:
    """Pinned local weights, head/tail WordPieces, eager float32 CPU inference."""

    def __init__(self, directory, manifest):
        self.directory = Path(directory)
        if manifest.get("repo_id") != ENCODER_ID or manifest.get("revision") != ENCODER_REVISION:
            raise ValueError("encoder identity differs")
        files = manifest.get("files")
        if not isinstance(files, dict) or len(files) != 11:
            raise ValueError("encoder requires the eleven pinned files")
        for name, expected in files.items():
            path = self.directory / name
            if path.is_symlink() or not path.is_file() or path.stat().st_size != expected["bytes"]:
                raise ValueError("encoder file missing or changed: " + name)
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
                raise ValueError("encoder file hash differs: " + name)
        for key, val in {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }.items():
            os.environ[key] = val
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.directory),
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
        if (self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id) != (101, 102, 0):
            raise ValueError("encoder special-token contract differs")
        self.model, loading = AutoModel.from_pretrained(
            str(self.directory),
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            dtype=torch.float32,
            attn_implementation="eager",
            output_loading_info=True,
        )
        self.compatibility = _validate_checkpoint_loading(self.model, loading, self.directory / "model.safetensors")
        self.model.to(device="cpu", dtype=torch.float32).eval().requires_grad_(False)

    def encode(self, values):
        """Callers freeze batch membership; deployment uses batches of one."""
        if not 1 <= len(values) <= 32:
            raise ValueError("encoder batch size must be 1..32")
        ids, masks = [], []
        for value in values:
            tokens = self.tokenizer.encode(
                input_document(value), add_special_tokens=False, truncation=False, verbose=False
            )
            if len(tokens) > 8_000_000:
                raise ValueError("input token count exceeds frozen limit")
            body = tokens if len(tokens) <= 254 else tokens[:127] + tokens[-127:]
            row = [101, *body, 102]
            ids.append(row + [0] * (256 - len(row)))
            masks.append([1] * len(row) + [0] * (256 - len(row)))
        torch = self.torch
        with torch.inference_mode(), torch.autocast(device_type="cpu", enabled=False):
            input_ids = torch.tensor(ids, dtype=torch.int64)
            mask = torch.tensor(masks, dtype=torch.int64)
            hidden = self.model(
                input_ids=input_ids, attention_mask=mask, token_type_ids=torch.zeros_like(input_ids), return_dict=True
            ).last_hidden_state
            expanded = mask.unsqueeze(-1).to(dtype=torch.float32)
            mean = (hidden.float() * expanded).sum(dim=1) / expanded.sum(dim=1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(mean, p=2, dim=1)
        return [validate_feature(row) for row in pooled.cpu().tolist()]


def _tokens(item):
    raw = contract.canonical_json({k: v for k, v in item.items() if k != "benchmark_id"})
    return frozenset(re.findall(r"\w+", raw.casefold()))


def _cosine(left, right):
    return len(left & right) / math.sqrt(len(left) * len(right)) if left and right else 0.0


class Predictor:
    """All adaptive state is reconstructed from the explicit evidence argument."""

    def __init__(self, config, *, asset_root=None, feature_provider=None):
        self.config = json.loads(canonical(config))
        if self.config.get("schema") != "five-candidate-model/v1" or self.config.get("arm") not in ARMS:
            raise ValueError("unsupported model declaration")
        self.arm = self.config["arm"]
        self.asset_root = Path(asset_root) if asset_root is not None else None
        self.provider = feature_provider
        self.encoder = None
        self.cache = OrderedDict()
        self.states = OrderedDict()
        self.counts = Counter()
        if self.arm in {"lexical_direct", "minilm_direct"}:
            expected_feature = "lexical384_v1" if self.arm == "lexical_direct" else "minilm384_head_tail_v1"
            if self.config.get("feature_kind", expected_feature) != expected_feature:
                raise ValueError("feature transform differs from declared arm")
            self.head = self.config.get("head", {})
            for key, n in (("beta", 385), ("center", 384)):
                values = self.head.get(key)
                if (
                    not isinstance(values, list)
                    or len(values) != n
                    or any(type(x) not in (int, float) or not math.isfinite(x) for x in values)
                ):
                    raise ValueError("direct head has malformed " + key)
            if self.head.get("status") not in {"CONVERGED", "BOUNDED_UNCONVERGED"}:
                raise ValueError("direct head must have completed a finite fit")
            if self.arm == "minilm_direct" and not self.config.get("encoder_manifest"):
                raise ValueError("MiniLM model must bind its encoder assets")

    def _base_logit(self, value):
        key = hashlib.sha256(input_document(value).encode()).hexdigest()
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        if self.provider is not None:
            feature = validate_feature(self.provider(value))
        elif self.arm == "lexical_direct":
            feature = lexical384(value)
        else:
            if self.asset_root is None:
                raise ValueError("MiniLM needs its owned asset root or a verified feature provider")
            if self.encoder is None:
                self.encoder = MiniLMEncoder(self.asset_root / "encoder", self.config["encoder_manifest"])
            feature = self.encoder.encode([value])[0]
        beta, center = self.head["beta"], self.head["center"]
        result = beta[0] + math.fsum(w * (x - c) for w, x, c in zip(beta[1:], feature, center, strict=True))
        if not math.isfinite(result):
            raise ValueError("non-finite direct logit")
        self.cache[key] = result
        if len(self.cache) > 8192:
            self.cache.popitem(last=False)
        return result

    def predict(self, input, labeled=None):
        evidence = contract.normalize_evidence(labeled)
        subject, item = contract.normalize_input(input, require_benchmark=bool(evidence))
        input_document(input)
        if evidence:
            key = contract.pair_key(input)
            own = [(value, int(label)) for value, label in evidence if contract.pair_key(value) == key]
        else:
            own = []
        if len(own) > 31:
            raise ValueError("same-pair evidence exceeds the 31-label horizon")
        self.counts["calls"] += 1
        if self.arm == "pm4":
            return (2.0 + math.fsum(y for _, y in own)) / (4 + len(own))
        if self.arm == "lexical_similarity":
            query = _tokens(item)
            weights = [0.1 + _cosine(query, _tokens(value[1])) ** 4 for value, _ in own]
            denominator = math.fsum(weights)
            estimate = math.fsum(w * y for w, (_, y) in zip(weights, own, strict=True)) / denominator if own else 0.0
            result = (2.0 + len(own) * estimate) / (4 + len(own))
            return float(max(0.0001, min(0.9999, result)))
        base = self._base_logit([subject, item])
        state_key = hashlib.sha256(canonical([[value, label] for value, label in own]).encode()).hexdigest()
        if state_key in self.states:
            fit = self.states[state_key]
            self.states.move_to_end(state_key)
        else:
            support = numerics.prepare_support(
                [self._base_logit(value) for value, _ in own],
                [(1.0,)] * len(own),
                [y for _, y in own],
                (0.0,),
            )
            fit = numerics.solve_map(support)
            self.counts["map_states"] += 1
            self.states[state_key] = fit
            if len(self.states) > 256:
                self.states.popitem(last=False)
        if fit.status != "OK":
            self.counts["fallback_calls"] += 1
            self.counts["fallback_" + fit.reason] += 1
            return 0.5
        return float(numerics.sigmoid(math.fsum((base, fit.mode[0]))))

    def diagnostics(self):
        return {
            "arm": self.arm,
            "counts": dict(self.counts),
            "fallback_calls": self.counts["fallback_calls"],
            "head_status": self.config.get("head", {}).get("status"),
        }


def make_predictor(config, asset_root=None, feature_provider=None):
    return Predictor(config, asset_root=asset_root, feature_provider=feature_provider)


def load_model(path, *, feature_provider=None):
    path = Path(path)
    if path.is_dir():
        path = path / "model.json"
    return make_predictor(json.loads(path.read_bytes()), path.parent, feature_provider)


def encode_inputs(values, feature_kind, encoder_root=None, *, encoder_manifest=None, batch_size=16):
    """Encode explicit visible inputs once; caller binds/cache-checks the rows.

    Batch membership is part of the numerical context and must be recorded.
    This routine does not receive, normalize, sample, or filter outcome labels.
    """
    if feature_kind == "lexical384_v1":
        return [list(lexical384(value)) for value in values]
    if feature_kind != "minilm384_head_tail_v1" or encoder_root is None:
        raise ValueError("unknown feature kind or absent encoder root")
    if type(batch_size) is not int or not 1 <= batch_size <= 32:
        raise ValueError("batch size must be 1..32")
    if encoder_manifest is None:
        if not __package__:
            raise ValueError("standalone encoding requires an explicit encoder manifest")
        from research.roadmap.features import model_identity

        encoder_manifest = model_identity()
    backend = MiniLMEncoder(encoder_root, encoder_manifest)
    result = []
    for begin in range(0, len(values), batch_size):
        result.extend(list(row) for row in backend.encode(values[begin : begin + batch_size]))
    return result


def uniform_acquisition(input, prediction=None, labeled=None, context=None):
    """Roadmap UTF-8 uniform hash rule; prediction does not affect decisions.

    This prospectively declared rule differs from the older profile's ASCII
    escaping on non-ASCII inputs and matches the new shared replay histories.
    """
    evidence = contract.normalize_evidence(labeled)
    contract.normalize_input(input, require_benchmark=bool(evidence))
    if not isinstance(context, dict):
        raise ValueError("streaming context is required")
    contract.canonical_json(context)
    remaining, items = context.get("labels_remaining"), context.get("items_remaining")
    if type(remaining) is not int or remaining < 0 or type(items) is not int or items <= 0:
        raise ValueError("invalid streaming counts")
    if remaining == 0:
        return False
    if items <= remaining:
        return True
    key = canonical([context, list(input)])
    uniform = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2**64
    return bool(uniform < remaining / items)
