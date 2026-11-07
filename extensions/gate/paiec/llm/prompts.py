"""Prompts and parsers of the LLM item-difficulty signal, verbatim from the validated runs. Runtime-safe (stdlib only).

Two prompts per item (docs/research/content.md, "End to end with the LLM prior"):
  rater     analysis/item_difficulty/llm_pilot.py SYSTEM / SCHEMA as used by analysis/content/llm_rate.py "generic":
            one JSON object; the signal is `difficulty_1_10`.
  selfeval  analysis/content/proxy/proxy_run.py "selfeval": one token, top-20 logprobs; the signal is
            P(No) / (P(Yes) + P(No)) to "would you answer this correctly?".
Item text follows analysis/content/e2e_llm/prepare.py: the full content if <= 6000 chars, else the within-benchmark
focus view (analysis/item_difficulty/views.py), prefixed by "[metadata: <item_features>]" when present.
tests/test_llm.py checks these strings against the analysis modules.
"""
import hashlib
import json
import math
import re

MAX_CHARS = 6000

RATER_SYSTEM = """You are an expert evaluator who estimates how hard benchmark tasks are for AI systems.
You will see one task exactly as it is given to the AI system being evaluated (possibly truncated; images and
attached files are not shown). Do not solve the task. Assess it and reply with ONE JSON object only."""

RATER_SCHEMA = """{
 "task_type": one of ["math_answer","math_proof","code_fix_repo","code_implement","doc_qa","web_gui_agent","knowledge_qa","other"],
 "domain": short string,
 "needs_tools_or_environment": true/false,   // must act in a repo / browser / GUI, run code
 "needs_unseen_inputs": true/false,           // depends on images, documents or files not shown in the text
 "n_reasoning_steps": integer,                // distinct non-trivial steps an expert would need
 "n_constraints": integer,                    // explicit conditions/requirements the answer must satisfy
 "expected_solution_size": integer,           // lines of code, or lines of derivation/answer
 "expert_minutes": number,                    // time for a skilled human expert
 "ambiguity": integer 0-3,                    // 0 = fully specified, 3 = underspecified / grader-dependent
 "difficulty_1_10": integer,                  // 1 trivial ... 10 among the hardest tasks of its kind
 "p_solve_frontier": number 0-1,              // probability a 2025 frontier model (agent if needed) gets it graded correct
 "p_solve_small": number 0-1                  // same for a small 7B open model
}"""

SELFEVAL_SYSTEM = "You are taking a test. Answer the task you are given as well as you can."
SELFEVAL_INSTR = ("Do not solve the task. Would you answer or complete this task correctly, as it would be graded? "
                  "Reply with exactly one word: Yes or No.")


def rater_messages(text):
    user = f"TASK (as given to the AI system):\n<<<\n{text}\n>>>\n\nReply with a JSON object with exactly these keys:\n{RATER_SCHEMA}"
    return [{"role": "system", "content": RATER_SYSTEM}, {"role": "user", "content": user}]


def selfeval_messages(text):
    return [{"role": "system", "content": SELFEVAL_SYSTEM},
            {"role": "user", "content": f"TASK:\n<<<\n{text}\n>>>\n\n{SELFEVAL_INSTR}"}]


def parse_json(text):
    """llm_pilot.parse: first {...} block, // comments stripped; None if unparseable."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    s = re.sub(r"//[^\n]*", "", m.group(0))
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def rater_value(obj):
    try:
        v = float((obj or {}).get("difficulty_1_10"))
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def p_no(top):
    """{token: logprob} -> P(No) / (P(Yes) + P(No)) over case/space variants (e2e_llm/build_signal.pno)."""
    y = sum(math.exp(v) for k, v in top.items() if k.strip().lower() == "yes")
    n = sum(math.exp(v) for k, v in top.items() if k.strip().lower() == "no")
    return n / (y + n) if y + n > 0 else None


def item_hash(item):
    """Item identity of the signal tables (e2e_llm/prepare.item_hash): features + content."""
    return hashlib.blake2b(((item.get("item_features") or "") + "\0" + (item.get("item_content") or "")).encode(),
                           digest_size=12).hexdigest()


# ---- focus view (analysis/item_difficulty/views.py) ----
def _norm(line):
    return " ".join(line.split()).lower()


def item_lines(content):
    return {_norm(l) for l in content.split("\n") if l.strip()}


def focus(c, df, n_items, n=2000, window=10, frac=0.0, min_df=2):
    """Keep lines rare within the benchmark (df <= max(min_df, frac*n_items)) and +-window neighbours."""
    lines = c.split("\n")
    thr = max(min_df, frac * n_items)
    rare = [bool(l.strip()) and df.get(_norm(l), 0) <= thr for l in lines]
    if not any(rare):
        return c[:n]
    keep = [False] * len(lines)
    for j, r in enumerate(rare):
        if r:
            for t in range(max(0, j - window), min(len(lines), j + window + 1)):
                keep[t] = True
    out = [l for j, l in enumerate(lines) if keep[j] and (rare[j] or df.get(_norm(l), 0) <= 0.5 * n_items)]
    s = "\n".join(out)
    return s[:n] if s.strip() else c[:n]


def item_text(item, df=None, n_items=0):
    """Prompt text of one item. `df` / `n_items`: line document frequency over the benchmark's items seen so far
    (at test time the pool is only partly visible; with an empty df the focus view is the first 6000 chars)."""
    c = item.get("item_content") or ""
    text = c if len(c) <= MAX_CHARS else focus(c, df or {}, n_items, MAX_CHARS)
    if item.get("item_features"):
        text = f"[metadata: {item['item_features']}]\n" + text
    return text
