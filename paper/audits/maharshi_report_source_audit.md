# Maharshi report source audit

## Target and scope

Audit class: code/repository review and report integration.

Target: `maharshi95/paiec-2026-report`, base commit `23a5fc8007c42950d3f0c96b7c0980b54ef8883f`, local snapshot `/workspace/scratch/2592d19f4bb5/work/paiec_repo_base`.

Inspected the full manuscript beginning, middle and end, README, the complete number generator, committed numerical macros and both generated tables. Inspected the active serialized configuration and relevant implementation in the exact `submit_gate.zip`. The report repository supplies manuscript and generated outputs; its scientific implementation and raw runs live in a separate repository according to README lines 3–5. The sibling `runs` directory is absent. A bounded GitHub metadata request for `maharshi95/paiec` returned HTTP 404 during this audit. The generator was read, rather than executed, because its inputs are unavailable.

Source pins:

- `main.tex` SHA256 `6dc4ac2240efe90af3fd9fbc911171c72cfa8b0dd603d0e6cb9d166f77692ecf`.
- `make_numbers.py` SHA256 `d0b27aaccf5956f3c9a56e4977376046cef315873070a3c62b1abe044557bd3c`.
- `generated/numbers.tex` SHA256 `182a879346c01e42543dc6567de0e900dde833abb4fa8558c827635df07c1a44`.
- `generated/tab_main.tex` SHA256 `d1954b7701065100447a3f87f6f9bdd5061ee2106ef4bd47943de33cf296aac5`.
- `generated/tab_ablation.tex` SHA256 `92c73ed1f7c49c952c5fcb992a1e99715cf411a4716d95e65eca107cd14dd040`.
- Exact deployment package `/workspace/scratch/2592d19f4bb5/evidence/gate/submit_gate.zip`, 190003 bytes, SHA256 `ddb2ad28e3887324f381777aedcefd75ec22515f09144c966f5df273c80b1a31`.

## Author metadata

The original title block names Maharshi Gor and University of Maryland, with placeholders for coauthors, affiliations, team and email (`main.tex:30–37`). The requested revision can place Ankit first while retaining Maharshi's verified name. Institutional affiliation for Ankit requires a separate supplied source.

## Actual model correspondence

The original headline model is Gate: `make_numbers.py:32–37` selects `gate_best_r30`, with `FINAL_USES_NETWORK=False`. The LLM difficulty prior is a separately configured variant. The package's active serialized class is `paiec.predictors.irt_mix:IRTItemMix`.

The manuscript describes hierarchical Rasch inference, global evidence within each benchmark, an online item map, subject regression, a VI/MAP gate and uncertainty acquisition (`main.tex:159–220`). The exact package supports these mechanisms:

| Mechanism | Manuscript | Exact package |
|---|---|---|
| Item map | 64 projected dimensions; variance shares 0.3/0.2/0.2/0.3 | `predictor.json` fields `rp_dim`, `rp_frac`, `kv_frac`, `dup_frac`; `irt_item.py:155–175` |
| Subject regression | Within-benchmark borrowing through subject descriptors | `mix_members` enables `lr_on` for both members; `irt_lr.py` and `irt_item.py` implement this |
| VI member | Variance factor 0.5 | `mix_members[0]`: `vi=true`, `vi_frac=0.5` |
| MAP member | Shared fitted state, separate online adaptation | `mix_members[1]`: `vi=false`; `irt_mix.py:55–81` |
| Gate | Own-label bins 0, 1, 2–3, 4–7, 8–15, 16+ | `irt_mix.py:24,94–108` |
| Gate weights | Rounded 2.36 and -1.71 after the first label | Exact weights 2.3581 and -1.7053; zero intercept; budget zero weights 1 and 0 |
| Prediction | Marginalization of logit uncertainty | Active `marginalize=true`; `irt_item.py:417–457` |
| Acquisition | Radius 0.30 and fill rule | `acquisition.json`; `acquisition/__init__.py:28–34,63–71` |

The exact positive-budget equation is `sigmoid(2.3581 * logit(p_VI) - 1.7053 * logit(p_MAP))`, after clipping each member probability to `[1e-4, 1-1e-4]`. This is an extrapolating logit combination, rather than a convex probability average. At zero own labels, it uses the clipped VI probability. All active fold-specific gate tables are null.

For the hierarchy equation, use conditional log-odds `eta = beta + delta - d`. The final predictive probability includes uncertainty marginalization before the outer gate. This preserves the distinction between latent conditional logits and returned predictions.

The original manuscript and deployment package match the same method/configuration family. A byte-level link from the deployment state to each evaluated run remains UNVERIFIED.

## Committed development results worth preserving

The following values are directly present in `generated/tab_main.tex:5–10`; the generator maps their run names in `make_numbers.py:22–44`.

| Development stack | Recorded hold-out ALC |
|---|---:|
| Global-evidence IRT | 0.2202 |
| Add online item map | 0.2154 |
| Add subject regression and VI | 0.2133 |
| Add logit gate | 0.2099 |
| Use acquisition radius 0.30 | 0.2077 |
| Separate LLM-prior variant | 0.2066 |

The stated panel contains 20 benchmarks, 654 subject–benchmark pairs and three seeds, or 1962 pair-seeds (`main.tex:115–130,459–490`; `generated/numbers.tex:38`). The final Gate row records Brier 0.259 at zero labels, 0.220 at one label, 0.200 at seven and 0.178 at 31. Seen/unseen ALC is 0.1977/0.2122 (`generated/numbers.tex:6–11`). These are archived generated results, distinct from a fresh replay in this session.

The recorded final-versus-IRT benchmark-mean paired difference is -0.0132 with a normal 95% interval half-width 0.0044, with improvements on all 20 benchmark means (`generated/numbers.tex:14–21`). The generator groups paired differences by benchmark and applies `1.96 * sample_sd / sqrt(number_of_benchmarks)` (`make_numbers.py:55–72`). Absolute ALC values are copied from `metrics.json`; their aggregation implementation is outside this report repository. Therefore preserve the benchmark-mean delta separately from subtraction of displayed absolute ALC values.

The manuscript discloses four looks at the hold-out during stack selection and gate fitting on an earlier development split (`main.tex:353–361`). A short attribution to the author's development protocol preserves the useful evidence without implying a fresh untouched test.

## Handwritten external Rasch–CAIMIRA comparison

`main.tex:447–455` describes a collaborator archive with frozen Rasch and rank-2 CAIMIRA, 384-dimensional hashed lexical item and subject features, pair-local MAP adaptation and random acquisition. The Gate comparison includes its radius-0.30 acquisition policy. Hence this compares full predictor/policy systems.

The manuscript records:

| Quantity | External Rasch–CAIMIRA | Gate |
|---|---:|---:|
| ALC on 19 benchmarks / 1905 pair-seeds | 0.2217 | 0.2068 |
| Brier at zero labels | 0.248 | 0.259 |
| Brier at 31 labels | 0.202 | 0.177 |

It reports a benchmark-clustered Gate-minus-external difference of -0.0139 ± 0.0072, and external wins on five named benchmarks. This comparison is absent from `make_numbers.py` and every generated table. Its values are handwritten. The manuscript supplies zero archive filename, SHA256, submission ID or run path for this comparator. Mapping it to `18-s44-rasch-caimira.zip` or submission 1033935 is UNVERIFIED.

The 19-benchmark subset excludes `devbench`: the manuscript reports 19 subjects with identical subject records, pooled labels exceeding 31 and an exception. The underlying failure log is unavailable. Retain this as an author-reported integration observation if needed; avoid attributing a reproduced failure to the exact selected candidate.

Its strongest useful implication is a development lead: transferable pair-local priors may complement broader within-benchmark adaptation. The joint mixture still needs its own matched empirical evaluation. The separate proposed 0.25 Rasch + 0.25 CAIMIRA + 0.50 Gate composition has only the previously recovered synthetic-validation record.

## Training and runtime boundaries

The manuscript's held-out evaluation fit uses 160 training benchmarks (`main.tex:46,475–477`). The package `build_info.json` declares 180 build benchmarks, containing all 20 named hold-out benchmarks; their set difference contains 160. This is consistent with a final deployment refit after development, but the exact fit receipt is unavailable. Describe the evaluation fit and deployment build separately.

The generator enforces coverage of all control keys `(subject_id, benchmark, seed)`. It leaves pool hashes, exact item membership, item cap, seed set and duplicate-key rejection unchecked (`make_numbers.py:48–72`). README lines 37–38 promise a stronger same-pool contract than this assertion establishes. This limits independent reproduction from the report repo alone.

Timing contexts differ: committed `AcqMs` is 17.7 from final-run metrics (`generated/numbers.tex:40`; `make_numbers.py:166`), while the handwritten platform-pattern benchmark reports 1.03 ms acquisition / 0.68 ms evaluation (`main.tex:528–531`). These can be different workloads. Preserve contextual labels if using either figure; current evidence supports neither a direct contradiction nor a fresh timing replication. The three-minute fit macro uses the maximum recorded fit time across the stack (`make_numbers.py:139,165`).

## Recommended integration

Keep the exact selected Rasch–CAIMIRA archive, hosted feedback and hash-matched local replay as the main empirical narrative. Close the report with the hierarchical Gate method and a compact attributed development stack table. A short appendix can preserve the separate 19-benchmark external comparison, its aggregation and acquisition context, and its unresolved archive identity. Keep the LLM-prior study separate from the active Gate deployment.

Suggested report-ready paragraph:

> The hierarchical extension has a complementary development record. Maharshi's archived report tables describe a 20-benchmark evaluation with 654 subject–benchmark pairs and three seeds. Global-evidence IRT scores 0.2202 ALC; adding the online item map, subject regression, VI, the logit gate and radius-0.30 acquisition reaches 0.2077. A separate 19-benchmark comparison reports stronger zero-label forecasts for an external Rasch–CAIMIRA model and stronger adaptation for Gate by budget 31. These observations motivate a composition that combines transferable priors with evidence shared across a new benchmark. The proposed joint mixture remains a separate experiment from the selected candidate's measured results.

## Final evidence verdict

Supported: original author metadata; method/configuration correspondence; committed generated values; generator formulas; precise handwritten comparison scope; deployment build declarations.

UNVERIFIED: raw replay values, exact comparator archive identity, full split/pool provenance, evaluated-fit-to-deployment-state lineage and a completed empirical evaluation of the joint mixture. Decisive next sources are `maharshi95/paiec/runs/`, the corresponding configs/manifests and the external comparator archive hash. This audit leaves the report repository unchanged.

## Scoped integration review

Reviewed the new `/workspace/scratch/2592d19f4bb5/work/paiec_repo/sections/page08.tex`, `appendix_shared.tex`, `appendix_extension.tex`, compact extension tables and `references.tex` after integration.

Passed: conditional-logit hierarchy; exact stored gate weights; member clipping and final marginalization; proposed 25/25/50 composition; current absence of a scored joint experiment; recorded original stack and ablation values; separate external-comparison scope and acquisition policies; commit-pinned reference. The preserved figure and original generated tables are byte-identical to the base snapshot. The generator's changes affect explanatory text and its output-header wording; computation remains unchanged.

Requested precision edits, delivered to the parent editor:

1. In the main extension-table caption, describe absolute ALC as stored aggregate metrics. Their full aggregation code lives outside the report repository.
2. Replace the phrase `180 fitted benchmarks` with `180 benchmarks in its build inventory` in the two extension appendices.
3. In the earlier-external-comparison paragraph, use `lower initial Brier score` and `lower endpoint Brier score` in place of `stronger ... Brier error`.
4. Optional clarity: the subject regression features include frozen ability/history summaries alongside subject descriptors, as implemented in `irt_lr.py:6–8,27,112–120`.

Also flagged stale README statements about inaccessible source and pending integration. The exact original manuscript is now incorporated, so those release-status statements need updating.

The three reviewed prose files contain zero instances of the prohibited standalone word, Unicode em dashes or TeX em-dash sequences. The reference inspection date, November 6 Pacific time, is consistent with the observed clock `2026-11-07 04:46:51 UTC`.

## Editor resolution

The integration adopts the requested precision edits: stored aggregate metrics in the caption; 180 benchmarks in the build inventory; lower Brier scores for comparisons; and frozen ability summaries in the subject regression. README and the source ledger now describe the completed manuscript integration and the remaining Overleaf connection gap.
