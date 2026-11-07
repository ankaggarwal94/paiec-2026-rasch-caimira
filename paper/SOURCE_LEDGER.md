# Evidence ledger for the release revision

Base: `maharshi95/paiec-2026-report` commit `8a156a203deb0fe3b2ab3e9e333803feb61c9501`. The author's Stanford affiliation and revised arrows are preserved.

The companion release is prepared for `ankaggarwal94/paiec-2026-rasch-caimira`. Paths below refer either to this paper source tree or that explicitly identified companion package. Public publication remains a final release step.

| Claim | Decisive artifact | Scope |
| --- | --- | --- |
| Selected method and original archive | Companion `runtime/original_submission/18-s44-rasch-caimira.zip`; `data/candidate_manifest.json` | 55,644 bytes; SHA-256 `23546eb90c7469fe7ea2704879775885f1b32e3277b4b7b3a1015aa1c7c6297b` |
| Equal probability blend, native CAIMIRA and offset Rasch | Exact `runtime/candidate/model.py`, top-level predict and branch setup | Active computation, distinct from unused shared machinery |
| 769 Rasch and 2,307 CAIMIRA coefficients | Two exact JSON state files and `audits/candidate_audit.json` | Centers excluded from fitted coefficient count |
| Formative ALC 0.160822, 40.27% endpoint Brier reduction | Supplied scoring HTML and screenshot; `data/feedback_by_budget.csv`, `feedback_pair_summary.csv`; `scripts/analyze_feedback.py` | Nine equally weighted subject–benchmark pairs; rounded summaries; 496 unique inputs |
| Local component ALC values | `data/local_replay_summary.json`; `scripts/analyze_local_replay.py` | Archived 153-pair/30-benchmark replay; one acquisition seed; label-informed selection |
| Exact CAIMIRA training lineage | Companion `data/provenance/submission.receipt.json`, `refit.json`, `freeze.json` | Earlier package's identical learned state; its offset policy stays separate from selected native adaptation |
| CAIMIRA state identity | SHA-256 `348088f33fb8080229f5475b95735719a43008a00af8a26b798831a138197eec` | Exact state equality |
| Completed refit | SHA-256 `d0ac3ada0487a14856a1077fabbc837f27f59acf7d3ec1c7367c134e5551220c` | Seed 17, CUDA, 300 Adam steps, 9,479,991 input rows, 10,951,005 trials, 1,345 pairs |
| Frozen corpus | SHA-256 `0b946c468d77aab6d3a8ad40a737a66155a16e3700d1875a86fdbe5a455ea2f7` | 44 benchmarks, 36 lineages, 176 projected source tables |
| Native fitting recipe and nuisance index | Companion `development/historical_source/research/broad_transfer/predictors.py`, `proxy_gpu/models.py`, `proxy_structured/structured.py` | Pair-weighted loss; nuisance effects indexed by unique serialized subject strings; learned tensors exported directly |
| Full training-source disclosure | Companion `data/sources/DATA_SOURCES.csv`, native/projected manifests; paper Appendix H | Forty native-table suites plus four curated supplements; source-specific rights and retrieval |
| Three sources shared with local replay | Companion `scripts/verify_provenance.py`, `data/training/local_replay_training_overlap.csv`, native table hashes | 25 replay pairs; exact evaluated-row intersection remains open |
| Fresh frozen inference | Companion `runtime/results/reference_smoke_results.json`, clean-environment receipts | Synthetic interface checks, deterministic output, guards; zero new accuracy evidence |
| Executable development code | Companion `development/develop.py`, 15-file source manifest, smoke summary | Two native synthetic fits and learned-state roundtrips; separate from a full corpus refit |
| Gate implementation | Companion `extensions/gate/submit_gate.zip`; `audits/gate_implementation_audit.md` | Exact archive SHA-256 `ddb2ad28e3887324f381777aedcefd75ec22515f09144c966f5df273c80b1a31`; proposed composition unscored |
| Hierarchical development results | Original generated tables/figure and `data/shared_report_manifest.json` | Preserved historical study; raw underlying runs remain outside recovered artifacts |
| Official requirements | [PAIEC competition page](https://aims.stanford.edu/competition) and author-supplied OpenReview form | Eight main pages; development/inference release; permitted additional data; source and assistance disclosure; PDF plus supplement ZIP at most 100 MB |

## Remaining evidence gaps

The exact historical Rasch fit receipt is located by a content-addressed archive manifest, while its byte retrieval remains pending. The complete projected corpus, some original extraction scripts, the raw local replay result CSV, and the hierarchical study's raw runs remain recovery targets. The release explicitly separates reproducible frozen inference and summary calculations from those broader historical reconstruction goals.

## Publication state

The existing paper repository is private. The connected GitHub capability can update it, while public repository creation and collaborator administration require the separate publishing helper or an account-side action. The revised paper marks the companion URL as prepared. Public release and replacement of the submitted OpenReview files remain visible steps in the compliance checklist.

## Current citation routing

All project artifact citations in the active paper resolve to the public companion repository. The unpublished hierarchical draft has zero bibliography entries. Exact earlier source identities remain in this historical ledger and the source manifests. The public target exists; artifact upload and collaborator acceptance remain separate completion steps.
