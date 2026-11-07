# The evaluator (`research/local_replay/`), what it honours and what it changes

A new, disclosed evaluator. It is a sibling of `research/recipe_eval` (never imported, never edited, so every frozen
Tier-B root keeps its implementation identity) and it is not "the proxy" (`research/hosted`, which scores recipes
arithmetically on hosted panels and executes no package).

## Protocol pieces taken from the organisers' kit, byte-verified

`kit.py` reads `tools/streaming_ingestion.py` and `tools/prediction_workers.py` out of
`research/hosted_runtime/kit/kit.zip` (SHA-256 `1b7a7f43…`, members checked against `KIT-MANIFEST.json`) and executes
them as modules. From them the evaluator uses, unchanged:

* `AcquisitionHook`: signature introspection that decides between the uniform rule (no `labeling.py`), streaming
  boolean hooks (prediction computed once, in the parent, only when the hook accepts it; keyword-only hooks such as the
  pm4 fallback's `(input, *, context=None)` are called with the keywords they accept; the arch-irt and bkp hooks are
  called with `prediction`, `labeled`, `context`) and one-argument legacy rank hooks; a non-`bool` return fails.
* `group_prediction_inputs` / `prediction_input_key`: canonical-JSON dedup of identical visible inputs per checkpoint.
* `parallel_predictions` and `prediction_concurrency`: fresh interpreters per checkpoint that import `model.py` once,
  receive the frozen evidence once and deep-copy it per call, one request in flight per worker, deadlines, process-group
  cleanup, and `prediction_config.json` lowering the worker count.

## Protocol pieces re-implemented from the README (the coordinator is not shipped)

`protocol.py`: the persistent 50/50 split with the odd item to acquisition, the fixed candidate order, the context
`{subject_id, benchmark_id, labels_remaining (31 - acquired), labels_acquired, max_labels 31, items_remaining}`, the
kit's uniform rule verbatim (`sha256(json.dumps([context, input], sort_keys=True, separators=(",", ":")))` with ASCII
escaping, first eight bytes over 2^64, below `labels_remaining / items_remaining`), nested budgets 0/1/3/7/15/31 with
all pairs finishing a checkpoint before any pair gets more labels, one global `labeled` list in pair-canonical order,
exhausted streams leaving later caps short (no error), every evaluation response row scored at every budget.

`scoring.py`: Brier over response rows, ten-bin count-weighted ECE, ALC weights 0.1/0.2/0.2/0.2/0.2/0.1, unweighted
pair mean; `pair_cell` is bitwise equal to `research/hosted/curves.pair_cell` (test), which is gate G1 of the emulator
against `research/roadmap/scoring.py`.

## Execution posture (`sandbox.py`, `runner.py`, `orchestrate.py`)

One package per `.venv/bin/python -I -B` interpreter; the zip is extracted to a scratch directory with path and size
checks; the environment is scrubbed (fake HOME, scratch TMPDIR, Hub/transformers offline, one BLAS thread, no CUDA);
outbound sockets are disabled in the parent before `model.py` is imported and in every kit worker through a
`sitecustomize` on the worker's startup path; the kit's own deadline, per-call timeout and process-group kill apply, and
the orchestrator adds a wall-clock kill. `model.py` and `labeling.py` are loaded by path with the package directory
first on `sys.path`, as the kit's tools do, without the smoke-test flag that makes the kit's HF templates skip loading.
Every result records the package SHA-256 and member list, the panel SHA-256, the kit SHA-256, interpreter and library
versions, worker count, call counts, label counts per pair and budget, and per-checkpoint wall time.

## Known differences from the hosted coordinator (disclosed)

* Local aliases (`local-<hash>`, `local_subject_<hash>`) instead of the hosted six-digit aliases: alias-keyed
  packages fall to their default table; the uniform rule's draws differ from any hosted draw (they are a function of
  the context strings) but are identical for every package here.
* The split and candidate order are hash ranks of (seed, benchmark, subject, item); the hosted PRNG is unknown.
* `labels_remaining` is always counted out of 31 (README example); the coordinator's exact context arithmetic is unseen.
* One response row per acquisition item is revealed (hash-chosen); the hosted choice among repeated runs is unseen.
* No 1,000-source-item draw: the whole frozen panel is one "draw", so each package sees 153 pairs at once and a global
  evidence list of up to 4,743 entries (hosted panels hold 8 to 9 pairs, so evidence lists are an order of magnitude
  shorter there); packages that scan the evidence list per call are correspondingly slower here, not different.
* Interpreter CPython 3.12.11 with torch 2.13.0 / transformers 4.57.6 on this Mac against the hosted Python 3.13 /
  torch 2.10.0 / transformers 5.5.0 baseline; see the Modal parity slice in `REPORT.md` for what that skew does.


