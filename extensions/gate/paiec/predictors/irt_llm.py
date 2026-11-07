"""LLM item-difficulty prior on top of the item stack: shift each item's prior difficulty mean by an LLM signal.

    d_i prior mean  +=  llm_c * llm_sd * s_i

- s_i: the standardised R+SE composite (llm/scorer.py): z(rater difficulty_1_10) + mean z(self-assessment P(No)),
  standardised within the benchmark. s_i = 0 (no shift) for items without a signal.
- llm_c = 0.156 is the pooled within-benchmark slope of z(d) on s, and llm_sd = 1.41 the median within-benchmark sd(d),
  both fitted on the content-protocol benchmarks (no hold-out benchmark). The prior variance is unchanged.
- Sources of s, first match:
    llm_table  a JSON {item_hash: s} computed offline (replays: analysis/content/e2e_llm/build_signal.py);
    llm_api    OpenAI calls at predict time with the validated prompts (llm/prompts.py), for the hosted run.
- Hold-out (split v2, Qwen3-32B / Qwen3.8-27B signals, table mode): item_lr_vi 0.2133 -> 0.2119, gate 0.2099 -> 0.2083
  (docs/results.md 2026-11-03). The OpenAI version uses the same prompts with one OpenAI model; it is not validated.

Runtime-only subclasses (FIT_AS = IRTItemLR): they reuse the item stack's cached fits, and llm_c = 0 reproduces the
parent exactly. Runtime: numpy + stdlib (urllib for the API).
"""
import json

from ..llm.prompts import item_hash
from .irt_item import IRTItemLR
from .irt_mix import IRTItemMix

_TABLES = {}
_MEMO = {}


class LLMPrior:
    _LLM_PARAMS = ("llm_c", "llm_sd", "llm_table", "llm_api", "llm_rater", "llm_selfeval", "llm_workers", "llm_wait_s",
                   "llm_timeout_s", "llm_max_items", "llm_deadline_h", "llm_shrink", "llm_stats", "llm_json_mode",
                   "llm_cache_dir")
    llm_c = 0.0
    llm_sd = 1.41
    llm_table = ""
    llm_api = False
    llm_rater = "gpt-4.1-mini"
    llm_selfeval = ("gpt-4.1-mini",)
    llm_workers = 16
    llm_wait_s = 30.0
    llm_timeout_s = 30.0
    llm_max_items = 20000
    llm_deadline_h = 5.0
    llm_shrink = 10.0
    llm_stats = None
    llm_json_mode = True
    llm_cache_dir = None

    def _llm_from_table(self, items):
        tab = _TABLES.get(self.llm_table)
        if tab is None:
            with open(self.llm_table) as f:
                tab = _TABLES[self.llm_table] = json.load(f)
        memo = _MEMO.setdefault(self.llm_table, {})
        out = []
        for it in items:
            k = (it.get("item_features", ""), it.get("item_content", ""))
            v = memo.get(k)
            if v is None:
                v = memo[k] = float(tab.get(item_hash(it), 0.0))
            out.append(v)
        return out

    def _llm_scorer(self):
        from ..llm.scorer import get_scorer
        return get_scorer(rater=self.llm_rater, selfeval=tuple(self.llm_selfeval or ()), workers=self.llm_workers,
                          wait_s=self.llm_wait_s, timeout_s=self.llm_timeout_s, max_items=self.llm_max_items,
                          deadline_h=self.llm_deadline_h, shrink=self.llm_shrink, stats=self.llm_stats,
                          json_mode=self.llm_json_mode, cache_dir=self.llm_cache_dir)

    def llm_scores(self, items):
        if not self.llm_c:
            return [0.0] * len(items)
        if self.llm_table:
            return self._llm_from_table(items)
        if self.llm_api:
            try:
                return [float(v) for v in self._llm_scorer().scores(items)]
            except Exception:  # noqa: BLE001 - the LLM prior is optional; fall back to the parent model
                return [0.0] * len(items)
        return [0.0] * len(items)

    def _info(self, items):
        out = super()._info(items)
        if not self.llm_c:
            return out
        shift = self.llm_c * self.llm_sd
        return [(x[0] + shift * s,) + tuple(x[1:]) if s else x for x, s in zip(out, self.llm_scores(items))]


class IRTItemLRLLM(LLMPrior, IRTItemLR):
    name = "irt_item_lr_llm"
    FIT_AS = IRTItemLR
    RUNTIME_PARAMS = IRTItemLR.RUNTIME_PARAMS + LLMPrior._LLM_PARAMS


class IRTItemMixLLM(LLMPrior, IRTItemMix):
    name = "irt_item_mix_llm"
    FIT_AS = IRTItemLR
    RUNTIME_PARAMS = IRTItemMix.RUNTIME_PARAMS + LLMPrior._LLM_PARAMS
