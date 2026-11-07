# The frozen local panel (2026-11-05)

Frozen 2026-11-05T03:21:03Z, before any package ran. `panel.json` (507,743,249 bytes, SHA-256
`7504b6678c99873ef248027e282779c455225c721231359b76ec7a651170b116`) lives only at
`~/workspace/cs321m-local-panel/panel/` because it contains gated measurement-db item text; `PANEL-MANIFEST.json`
(here) binds that hash, the source-table hashes and the counts, and `pairs.csv` (here) lists every pair without
content. Every package run verifies the panel hash against the manifest before it starts.

## 1. Source

`aims-foundations/measurement-db` at revision `54074453da7b9911be0f48e4c560be547bc84d21` (the revision the
selected entry trained on), read through the project's Hub login, four Parquet tables per benchmark
(`benchmarks`, `subjects`, `items`, `responses`), downloaded with `huggingface_hub` into the workspace cache
(SHA-256 of every table consumed is in the manifest). No raw archive, trace or asset was fetched.

## 2. Eligibility (why each benchmark is in or out)

Gate, in order, applied to all 281 directories of the revision:

1. Catalog (`benchmarks.parquet`): `release == public`, `granularity == item`, binary 0/1 response scale.
2. Footer counts: at least 80 items and at least 1 subject row.
3. Not one of the 195 benchmarks in `handoffs/device2-training-folds-20261103/benchmarks-195.txt` (the arch-irt
   training list, which also contains the six organiser-public validation families).
4. Not one of the structured (s44) lineage's panels, whose state files carry no provenance: `cellverse`, `bfcl`,
   `livecodebench`, `swebench`, `abstract_reason`; nor `scale_mrt` (excluded by the paiec data configuration).
5. Per pair: at least 80 distinct items with a recorded 0/1 response; null responses are dropped as missing labels;
   a subject with ANY non-binary numeric response in the benchmark is excluded (no binarisation anywhere).

Of the 86 directories outside the 195, the strict catalog-binary gate leaves 16 public benchmarks with two or more
qualifying subjects and 10 with exactly one. Because the brief asks for at least 25 benchmarks, three choices widen the
pool and are disclosed as tiers:

| tier | rule | benchmarks | pairs |
|---|---|---|---|
| S (strict, multi-subject) | catalog `binary`, >= 2 qualifying subjects | finegrain_t2i, hallusionbench, hle, llmail_inject, mathconstraint, medhelm_ehr_sql, morphkv, naturebench, nester, ood_prediction, pku_saferlhf, prox, revolve, software_world_models, tau2_bench, wmt_mqm (16) | 116 |
| 1 (strict, single subject) | catalog `binary`, exactly 1 qualifying subject | aegis, alpha_sql, averimatec, enginemt_qa, felm, igakuqa119, llm_bp_tag, mind2web, proofaug, sugarcrepe (10) | 10 |
| M (catalog `mixed`, data binary) | catalog response type `mixed`, every admitted subject's recorded responses are exactly 0/1, and the construct is a per-item verdict on a language-model or agent run | dtap_bench (task-utility and attacker-success verdicts), medagentboard (medical QA and workflow assessments), mergebench (safety assessment records), multi_moe (multiple-choice correctness) (4) | 27 |

Tier 1 breaks the brief's "at least 2 subjects" letter to serve its intent (difficulty is a benchmark property, so
benchmark count carries the effective sample size); tier M breaks the catalog letter of "binary" while keeping the
data binary. Readings are reported on all 30 benchmarks and, as sensitivity, on tier S alone.

Rejected although their recorded values are 0/1: `tabarena` (tabular machine-learning methods on table rows, not a
language-model evaluation) and `mlip_arena` (interatomic potentials), `helm_bold` (fraction of toxic completions,
catalog `fraction`), `preference_dissection` (nominal preference labels), `intercode` and `livebench` (only 1 and 3
of 26 and 200 subjects are binary-only; the benchmarks are reward- and score-valued). Rejected on data grounds:
`mmmu_dev_val`, `chartmuseum`, `ineqmath`, `naturalreasoning` (every response null at this revision).

Two directional notes. `llmail_inject` and `wmt_mqm` are catalog-binary with `lower_is_better` (1 = attack succeeded,
1 = error annotated); the kit scores P(response = 1) symmetrically, so they enter unchanged, and both contribute
near-deterministic pairs (base rates near 0), which the hosted law needs.

## 3. Draw (selection.json, seed 20261105)

Per benchmark at most 10 subjects. Pairs were assigned to four strata by the pair's base rate over all its labelled
rows, e = |0.5 - a|: very hard e >= 0.40, hard 0.288 <= e < 0.40, mid 0.10 <= e < 0.288, coin e < 0.10, with target
shares 16 / 20 / 44 / 20 % taken from the hosted law (36 % of hosted pairs beyond e 0.288, 16 % beyond 0.40, 20 % within
0.10 of a coin flip; reviews/codabench-scoring-analysis-20261027 section 3.5 and reviews/hosted-results-robustness-20261104
section 2). Strata were filled in that order, each time taking the benchmark with the fewest selected pairs (random
ties) and a random eligible pair of that benchmark in the stratum. With `--n-target 200` the pool under the caps is
exhausted: 153 pairs over 30 benchmarks (strata filled 31 / 36 / 56 / 30; mid and coin ran dry first). Selecting pairs
by their base rate uses labels for panel design, not for any package; it is disclosed here and frozen before any run.

Achieved distribution (evaluation-pool base rates): mean u 0.1661 (hosted law 0.157 +- 0.020, i.e. 0.45 SD above),
SD across pairs 0.078; share of pairs with e >= 0.288: 0.438 (hosted 0.36); e >= 0.40: 0.203 (0.16); e < 0.10: 0.196
(0.20). The panel is slightly harder than the hosted average, mostly through `llmail_inject` and `wmt_mqm`; the report
therefore also reads every contrast by difficulty tertile and at u 0.157 by regression.

## 4. Per-pair protocol (kit README, section "Streaming test-time adaptation")

* Items: all distinct items of the pair with a 0/1 response. A pair with more than 200 items keeps the 200 lowest
  `sha256("cap", split_seed, benchmark, subject_id, item_id)` ranks (so at most 100 evaluation items, the paiec
  `max_eval_items_per_pair` convention, and at most 100 acquisition candidates).
* Persistent 50/50 split: items ranked by `sha256("split", split_seed, benchmark, subject_id, item_id)`; the first
  ceil(m/2) form the acquisition pool, the rest the evaluation pool, so an odd count gives the extra item to
  acquisition. The rule is per item, so adding items later would balance the pools without moving old ones.
* Acquisition stream: the acquisition items in `sha256("order", order_seed, ...)` rank; one recorded response per item is
  the label revealed (row index from `sha256("reveal", ...)` when an item has several runs); every package sees the same
  stream, the same labels and the same evaluation rows.
* Evaluation: every response row of every evaluation item is scored at every budget with the probability predicted for
  its visible input (`item_content`, `item_features`, response-level `interactors`, local benchmark alias
  `local-<sha256(slug)[:20]>`); identical visible inputs are predicted once (the kit's canonical-JSON dedup).
* Subject fields: the eight kit strings from `subjects.parquet`, blanks as `""`; the context `subject_id` is a local
  alias (`local_subject_<hash>`), which, like the benchmark alias, differs from the hosted six-digit aliases by design.

Totals: 14,494 acquisition items, 14,492 distinct evaluation inputs, 21,788 evaluation response rows (repeated runs
are concentrated in llmail_inject, tau2_bench, mathconstraint, wmt_mqm and revolve). Item text is long on two
benchmarks (mind2web mean 197 kB, software_world_models mean 185 kB per item; tau2_bench 20 kB), which stresses
packages the way long hosted items would; nothing is truncated by the evaluator.

## 5. What this panel is not

Not the hidden hosted panel (different benchmarks, aliases, subjects, items and draw rule); not alias-matched (keyed
packages fall to their default table); not difficulty-identical (u 0.166 against 0.157); and not independent across
pairs within a benchmark (readings therefore carry benchmark-cluster standard errors).

## 6. Sub-panel: one subject per benchmark (added 2026-11-05T08:47Z, before any reading of it)

Hosted panels hold 8 to 9 pairs over 7 benchmarks with, in every observed panel, one subject per benchmark except one
benchmark carrying two. The frozen panel allows up to 10 subjects per benchmark, which hands a package that pools
evidence across subjects of one benchmark (the arch-irt lineage's per-benchmark item amortisation, for instance) up to
ten times the cross-subject evidence it would see hosted. `panel-max1.json` (SHA-256 `a29c3aae…`, manifest
`PANEL-MAX1-MANIFEST.json`) keeps the first pair of every benchmark in frozen panel order (pairs are sorted by subject id
within a benchmark, a rule blind to outcomes): 30 pairs over the same 30 benchmarks, evaluation-pool u mean 0.172,
2,921 distinct evaluation inputs, 3,992 response rows, identical splits, streams, labels and rows for the kept pairs.
Every package is run on it as a second reading; with 30 pairs its resolution is coarser (roughly 0.01), so it is read
as a direction check on the sharing effect, not as a second tiering.


