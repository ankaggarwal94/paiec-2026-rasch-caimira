"""Minimal OpenAI Chat Completions client (stdlib urllib; the hosted image cannot pip install the SDK).

The competition allows runtime calls to OpenAI's inference API with your own key (docs/competition.md). The key is
read from $OPENAI_API_KEY, else from a file `openai_api_key` in the submission root (scripts/build_submission.py
--openai-key-file copies it there; never commit it), else from `submission_config.json` {"api_keys": {"OPENAI_API_KEY"}}
as in the organizers' BLE baseline.
"""
import json
import os
import random
import time
import urllib.error
import urllib.request

from . import prompts

API_URL = "https://api.openai.com/v1/chat/completions"
KEY_FILE = "openai_api_key"


def find_key(*dirs):
    k = os.environ.get("OPENAI_API_KEY", "").strip()
    if k:
        return k
    for d in (*dirs, os.getcwd()):
        if not d:
            continue
        p = os.path.join(d, KEY_FILE)
        if os.path.isfile(p):
            with open(p) as f:
                k = f.read().strip()
            if k:
                return k
        p = os.path.join(d, "submission_config.json")   # the organizers' BLE baseline layout: {"api_keys": {...}}
        if os.path.isfile(p):
            try:
                with open(p) as f:
                    k = ((json.load(f).get("api_keys") or {}).get("OPENAI_API_KEY") or "").strip()
            except (OSError, ValueError, AttributeError):
                k = ""
            if k:
                return k
    return None


class OpenAIError(RuntimeError):
    pass


class OpenAIChat:
    def __init__(self, key, url=None, timeout=30.0, retries=3):
        self.key, self.timeout, self.retries = key, float(timeout), int(retries)
        self.url = url or os.environ.get("OPENAI_CHAT_URL", API_URL)

    def complete(self, body):
        data = json.dumps(body).encode()
        last = None
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(self.url, data=data, method="POST", headers={
                "Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read().decode())
            except urllib.error.HTTPError as e:
                last = OpenAIError(f"HTTP {e.code}: {e.read()[:300]!r}")
                if e.code not in (408, 409, 429) and e.code < 500:
                    raise last from None          # bad request / auth: retrying will not help
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
                last = OpenAIError(repr(e))
            if attempt < self.retries:
                time.sleep(min(2 ** attempt, 8) * (0.5 + random.random()))
        raise last

    def rate(self, model, text, json_mode=True):
        """Rater prompt -> (difficulty_1_10 or None, parsed JSON). Greedy, <= 400 output tokens (as the vLLM runs)."""
        body = {"model": model, "messages": prompts.rater_messages(text), "temperature": 0,
                "max_completion_tokens": 400, "seed": 0}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        out = self.complete(body)
        obj = prompts.parse_json(out["choices"][0]["message"].get("content") or "")
        return prompts.rater_value(obj), obj

    def selfeval(self, model, text):
        """Self-assessment prompt -> (P(No) or None, {token: logprob} of the first token). One greedy token, top 20."""
        body = {"model": model, "messages": prompts.selfeval_messages(text), "temperature": 0,
                "max_completion_tokens": 1, "logprobs": True, "top_logprobs": 20, "seed": 0}
        out = self.complete(body)
        content = ((out["choices"][0].get("logprobs") or {}).get("content") or [])
        top = {t["token"]: round(float(t["logprob"]), 4) for t in content[0].get("top_logprobs", [])} if content else {}
        return prompts.p_no(top), top
