"""Runtime LLM item signal: OpenAI calls, a shared disk cache, and per-benchmark standardisation.

    s_i = standardise_b( z_b(R_i) + mean_m z_b(P_m,i) )

R = rater `difficulty_1_10`, P_m = self-assessment P(No) of model m (prompts.py). This is the R+SE composite of
analysis/content/e2e_llm/build_signal.py with the Qwen models replaced by OpenAI models. Offline, z_b used the whole
benchmark's eval items. At test time only the items seen so far are known, so every mean / sd is a running estimate
over the benchmark's scored items, shrunk towards fixed priors with `shrink` pseudo-items. The priors are the
content-protocol (non-hold-out) Qwen3-32B values; rerun `scripts/llm_openai.py stats` on OpenAI outputs to refresh
them. An item's s is frozen the first time it is computed in a process, so predictions stay stable within a worker.

Robustness (an LLM failure must never fail the run): no key -> disabled; a failed or slow call -> the item gets
s = 0 (= the model without the LLM prior); `max_fail` consecutive failures open a circuit breaker for the process;
no new calls after `deadline_h` hours from the first call of the run (a marker file in the cache dir) or once
`max_items` items are scored. Calls run on a thread pool; a predict blocks at most `wait_s` for its items, and
calls still in flight finish in the background and land in the cache.
"""
import json
import math
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait as fwait

from . import prompts
from .openai_client import OpenAIChat, find_key

# Qwen3-32B on the content protocol (12,580 items, 137 non-hold-out benchmarks): global mean, pooled within-benchmark sd
DEFAULT_STATS = {"R": [4.80, 1.09], "P": [0.26, 0.19], "S": [0.0, 1.49]}
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # submission root (or src/)

_SCORERS = {}
_LOCK = threading.Lock()


def get_scorer(**cfg):
    key = json.dumps(cfg, sort_keys=True, default=str)
    with _LOCK:
        if key not in _SCORERS:
            _SCORERS[key] = LLMScorer(**cfg)
        return _SCORERS[key]


def _log(msg):
    print(f"[paiec.llm] {msg}", file=sys.stderr, flush=True)


def _cache_dir(path):
    sub = ROOT if os.path.exists(os.path.join(ROOT, "model.py")) else None    # a submission dir, not the repo's src/
    for d in (path, os.environ.get("PAIEC_LLM_CACHE_DIR"), sub, tempfile.gettempdir()):
        if not d:
            continue
        try:
            os.makedirs(d, exist_ok=True)
            probe = os.path.join(d, ".paiec_llm_probe")
            with open(probe, "a"):
                pass
            return d
        except OSError:
            continue
    return None


class _Running:
    """Shrunk running mean / sd of a stream of values."""
    __slots__ = ("n", "s1", "s2")

    def __init__(self):
        self.n, self.s1, self.s2 = 0, 0.0, 0.0

    def add(self, x):
        self.n += 1
        self.s1 += x
        self.s2 += x * x

    def get(self, prior, k):
        m0, sd0 = prior
        n = self.n
        mean = (self.s1 + k * m0) / (n + k)
        ss = max(self.s2 - 2 * mean * self.s1 + n * mean * mean, 0.0)   # sum (x - mean)^2 over the data
        sd = math.sqrt((ss + k * sd0 * sd0) / (n + k))
        return mean, sd if sd > 1e-9 else sd0


class LLMScorer:
    def __init__(self, rater="gpt-4.1-mini", selfeval=("gpt-4.1-mini",), workers=16, wait_s=30.0, timeout_s=30.0,
                 retries=3, max_items=20000, deadline_h=5.0, max_fail=20, shrink=10.0, stats=None, json_mode=True,
                 cache_dir=None, client=None):
        self.rater, self.selfeval = rater, tuple(selfeval or ())
        self.wait_s, self.max_items, self.deadline_h = float(wait_s), int(max_items), float(deadline_h)
        self.max_fail, self.shrink, self.json_mode = int(max_fail), float(shrink), bool(json_mode)
        self.stats = {**DEFAULT_STATS, **(stats or {})}
        self.lock = threading.RLock()
        self.vals = {}          # (hash, kind, model) -> float | None
        self.bench_of = {}      # hash -> benchmark id
        self.by_bench = {}      # bench -> [hash] (scored items, in arrival order)
        self.df = {}            # bench -> {line: count}, over items seen (focus view of long items)
        self.seen_lines = {}    # bench -> set(hash) already counted in df
        self.frozen = {}        # hash -> s
        self.inflight = {}      # (hash, kind, model) -> future
        self.fails = 0
        self.disabled = None
        self.client = client
        if self.client is None:
            key = find_key(ROOT)
            if key:
                self.client = OpenAIChat(key, timeout=timeout_s, retries=retries)
            else:
                self.disabled = "no OpenAI API key"
        d = _cache_dir(cache_dir)
        self.cache_path = os.path.join(d, "paiec_llm_cache.jsonl") if d else None
        self._offset = 0
        self._load_cache()
        self.t0 = self._start_time(d)
        self.pool = ThreadPoolExecutor(max_workers=int(workers), thread_name_prefix="paiec-llm") if self.client else None
        if self.disabled:
            _log(f"disabled: {self.disabled}")

    # ---- cache ----
    @staticmethod
    def _start_time(d):
        if not d:
            return time.time()
        p = os.path.join(d, "paiec_llm_start")
        try:
            with open(p, "x") as f:
                f.write(str(time.time()))
        except FileExistsError:
            pass
        except OSError:
            return time.time()
        try:
            with open(p) as f:
                t0 = float(f.read().strip() or time.time())
        except (OSError, ValueError):
            return time.time()
        if time.time() - t0 > 8 * 3600:      # older than a hosted run can last: a stale marker from an earlier run
            t0 = time.time()
            try:
                with open(p, "w") as f:
                    f.write(str(t0))
            except OSError:
                pass
        return t0

    def _load_cache(self):
        """Read entries appended since the last read (other workers share the file)."""
        if not self.cache_path or not os.path.exists(self.cache_path):
            return
        try:
            with open(self.cache_path, "rb") as f:
                f.seek(self._offset)
                chunk = f.read()
        except OSError:
            return
        end = chunk.rfind(b"\n") + 1      # complete lines only
        self._offset += end
        with self.lock:
            for line in chunk[:end].splitlines():
                try:
                    r = json.loads(line)
                    self._put(r["h"], r["b"], r["k"], r["m"], r["v"])
                except (ValueError, KeyError, TypeError):
                    continue

    def _append(self, rec):
        if not self.cache_path:
            return
        try:
            with open(self.cache_path, "a") as f:     # one short line per write: atomic under O_APPEND
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass

    def _put(self, h, b, kind, model, v):
        k = (h, kind, model)
        if k in self.vals and self.vals[k] is not None:
            return
        self.vals[k] = v
        if h not in self.bench_of:
            self.bench_of[h] = b
            self.by_bench.setdefault(b, []).append(h)

    # ---- fetching ----
    def _tasks(self):
        return [("R", self.rater)] * bool(self.rater) + [("P", m) for m in self.selfeval]

    def _can_call(self):
        if self.disabled or self.client is None:
            return False
        if self.fails >= self.max_fail:
            self.disabled = f"{self.fails} consecutive API failures"
            _log(f"disabled: {self.disabled}")
            return False
        if time.time() - self.t0 > self.deadline_h * 3600:
            self.disabled = f"deadline of {self.deadline_h} h reached"
            _log(f"disabled: {self.disabled}")
            return False
        if len(self.bench_of) >= self.max_items:
            self.disabled = f"max_items {self.max_items} reached"
            _log(f"disabled: {self.disabled}")
            return False
        return True

    def _call(self, h, b, kind, model, text):
        try:
            if kind == "R":
                v, _ = self.client.rate(model, text, self.json_mode)
            else:
                v, _ = self.client.selfeval(model, text)
            with self.lock:
                self.fails = 0
                self._put(h, b, kind, model, v)
            self._append({"h": h, "b": b, "k": kind, "m": model, "v": v})
        except Exception as e:  # noqa: BLE001 - never propagate into predict
            with self.lock:
                self.fails += 1
                if self.fails in (1, self.max_fail):
                    _log(f"call failed ({self.fails} in a row): {e!r}")
        finally:
            with self.lock:
                self.inflight.pop((h, kind, model), None)

    def _observe_lines(self, b, h, item):
        if h in self.seen_lines.setdefault(b, set()):
            return
        self.seen_lines[b].add(h)
        df = self.df.setdefault(b, {})
        for line in prompts.item_lines(item.get("item_content") or ""):
            df[line] = df.get(line, 0) + 1

    def fetch(self, items):
        """Make sure every item has its raw signals (or a pending call); block up to wait_s for them."""
        recs = []
        for it in items:
            h = prompts.item_hash(it)
            recs.append((h, it.get("benchmark_id", ""), it))
        with self.lock:
            missing = [(h, b, it) for h, b, it in recs if any((h, k, m) not in self.vals for k, m in self._tasks())]
        if not missing:
            return
        self._load_cache()
        futs = []
        with self.lock:
            for h, b, it in missing:
                self._observe_lines(b, h, it)
            for h, b, it in missing:
                todo = [(k, m) for k, m in self._tasks() if (h, k, m) not in self.vals]
                if not todo:
                    continue
                for k, m in todo:
                    f = self.inflight.get((h, k, m))
                    if f is None:
                        if not self._can_call():
                            continue
                        text = prompts.item_text(it, self.df.get(b), len(self.seen_lines.get(b, ())))
                        f = self.inflight[(h, k, m)] = self.pool.submit(self._call, h, b, k, m, text)
                    futs.append(f)
        if futs and self.wait_s > 0:
            fwait(futs, timeout=self.wait_s)

    # ---- standardisation ----
    def _composite(self, b):
        """Per-benchmark z functions from the scored items so far -> (s_raw of each hash, its mean / sd)."""
        k = self.shrink
        hs = self.by_bench.get(b, [])
        comp = {}
        for kind, model in self._tasks():
            r = _Running()
            for h in hs:
                v = self.vals.get((h, kind, model))
                if v is not None:
                    r.add(v)
            comp[(kind, model)] = r.get(self.stats[kind], k)
        def raw(h):
            zr = None
            if self.rater:
                v = self.vals.get((h, "R", self.rater))
                if v is not None:
                    m, sd = comp[("R", self.rater)]
                    zr = (v - m) / sd
            zp = []
            for model in self.selfeval:
                v = self.vals.get((h, "P", model))
                if v is not None:
                    m, sd = comp[("P", model)]
                    zp.append((v - m) / sd)
            parts = [x for x in (zr, sum(zp) / len(zp) if zp else None) if x is not None]
            return sum(parts) if parts else None
        rs = _Running()
        out = {}
        for h in hs:
            x = raw(h)
            out[h] = x
            if x is not None:
                rs.add(x)
        return out, rs.get(self.stats["S"], k)

    def scores(self, items):
        """Standardised composite per item; 0.0 where no signal is available (= no LLM prior for that item)."""
        self.fetch(items)
        out, cache = [], {}
        with self.lock:
            for it in items:
                h = prompts.item_hash(it)
                if h in self.frozen:
                    out.append(self.frozen[h])
                    continue
                b = self.bench_of.get(h, it.get("benchmark_id", ""))
                if b not in cache:
                    cache[b] = self._composite(b)
                raw, (m, sd) = cache[b]
                x = raw.get(h)
                if x is None:
                    out.append(0.0)
                    continue
                s = self.frozen[h] = (x - m) / sd
                out.append(s)
        return out
