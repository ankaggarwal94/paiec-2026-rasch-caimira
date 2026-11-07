"""Canonical model identity from (visible) subject fields.

Raw names differ wildly across benchmarks: "Anthropic Claude 3.5 Sonnet", "claude-3-5-sonnet-20241022",
"anthropic/claude-3.5-sonnet", "GPT_4O_2024_08_06", "o3-mini-high". The canonical form is the sorted set of
alphanumeric tokens after dropping org prefixes, dates, active-parameter tags (a35b) and run-mode words
(thinking/instruct/high/...). Mode words are dropped on purpose: this is *model* identity for sharing
information across benchmarks, while the exact subject (incl. harness / effort) stays `subject_key`.
"""
import json
import re

from .hashing import parse_kv

ORG = {"openai", "anthropic", "google", "meta", "alibaba", "moonshot", "moonshotai", "bytedance", "amazon",
       "mistralai", "microsoft", "nvidia", "xai", "opengvlab", "ai", "inc", "thudm", "zhipuai", "tiiuae",
       "huggingface", "together", "togethercomputer", "fireworks", "openrouter", "vertex", "azure", "bedrock",
       "accounts", "models", "deepseekai", "qwenlm", "llamaindex", "ollama"}
DROP = {"thinking", "think", "instruct", "chat", "preview", "latest", "exp", "it", "hf", "high", "low", "medium",
        "xhigh", "minimal", "reasoning", "model", "fp8", "awq", "gguf", "bf16", "fp16", "int4", "int8", "api",
        "default", "the", "v"}
_DATE = re.compile(r"(19|20)\d\d|\d{6,8}|(0[1-9]|1[0-2])[0-3]\d")
_ACTIVE = re.compile(r"a\d+(\.\d+)?b")
_FULLDATE = re.compile(r"(19|20)\d\d[-_/.]?(0[1-9]|1[0-2])[-_/.]?[0-3]\d")
# keys in subject_features_extra that carry a model name (visible at test time)
EXTRA_NAME_KEYS = ("source_model_name", "source_model_label", "model_config", "source_model_filename", "model",
                   "model_name")


def canonical(name: str) -> str:
    toks = set()
    for t in re.split(r"[^a-z0-9]+", _FULLDATE.sub(" ", name.lower())):
        if not t or t in ORG or t in DROP or _DATE.fullmatch(t) or _ACTIVE.fullmatch(t):
            continue
        toks.add(t)
    return " ".join(sorted(toks))


def name_candidates(subject: dict, include_display: bool = False) -> list:
    """Model-name strings available in a subject dict (display_name only when include_display)."""
    out = [subject.get("normalized_name", "")]
    extra = subject.get("subject_features_extra", "") or ""
    kv = parse_kv(extra)
    out += [kv.get(k, "") for k in EXTRA_NAME_KEYS]
    gs = kv.get("generation_settings", "")
    if gs.startswith("{"):
        try:
            out.append(str(json.loads(gs).get("model", "")))
        except ValueError:
            pass
    if include_display:
        out.append(subject.get("display_name", "") or "")
    return [n for n in out if n and n.lower() != "none"]


def model_identity(subject: dict, include_display: bool = False) -> str:
    """Canonical identity: the first candidate with a non-empty canonical form ('' if none)."""
    for n in name_candidates(subject, include_display):
        c = canonical(n)
        if c:
            return c
    return ""


def identity_set(subject: dict, include_display: bool = False) -> set:
    return {c for c in (canonical(n) for n in name_candidates(subject, include_display)) if c}
