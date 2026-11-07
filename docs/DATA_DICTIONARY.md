# Data dictionary and measurement scope

## Hosted feedback

`data/hosted/feedback_by_budget.csv` contains 54 rows: nine subject–benchmark pairs at six budgets. Fields are selected submission ID, anonymous subject ID, anonymous benchmark ID, count of unique evaluation inputs, label budget, Brier error, and expected calibration error. Decimal scores are transcribed at the six-place precision supplied by Codabench.

`feedback_pair_summary.csv` contains the nine displayed pair ALCs and derived endpoint gains. `computed_metrics.json` records aggregate calculations, rounding checks, and denominators. ALC weights are 0.1, 0.2, 0.2, 0.2, 0.2, and 0.1 at budgets 0, 1, 3, 7, 15, and 31. Pair means receive equal weight. The 496 distinct evaluated inputs are distinct from the response count used inside a pair's score.

The original HTML includes a temporary download link. The release preserves its score contents through the transcription and omits that expiring credential-bearing link.

## Archived component replay

`data/local/summary.json` preserves aggregate curves and selected execution receipts for Rasch, the blend, and CAIMIRA on 153 pairs, 30 benchmarks, and 21,788 response rows. `pairs.csv` records the historical population. `panel_manifest.json` identifies source tables, split/order seed, and frozen panel digest. The same acquisition/evaluation population underlies all three selected package records.

`data/training/local_replay_training_overlap.csv` compares those 30 benchmark names with the 44-benchmark CAIMIRA corpus and verifies native table identity for three shared sources. It identifies 25 shared-source pairs. Wider ancestry and evaluated-row intersections require the underlying inputs. The raw result CSV and per-response predictions remain outside this prepared package; aggregate-derived confidence intervals would imply unavailable replication evidence.

## Training corpus

`data/training/training_benchmarks.csv` lists all 44 suites, source-lineage groups, projected directories, and the four projected-table digests per suite. `projected_training_tables.csv` enumerates all 176 paths and expected SHA-256 values.

`data/provenance/freeze.json`, `refit.json`, and `submission.receipt.json` retain the archived evidence chain for the exact CAIMIRA state. The earlier receipt belongs to a scalar-offset CAIMIRA package; only its frozen state and fitting provenance transfer to the selected native-adaptation branch.

`data/sources/` distinguishes publisher metadata, historical revisions, source download paths, and redistribution terms. The native source tables differ from the derived `facts/items/subjects/ungraded` tables consumed by the historical trainer. A verified upstream download alone leaves the projection step to the original extraction pipeline.

## Synthetic fixtures and extension results

Synthetic runtime and training fixtures are generated interface examples. They carry zero real-benchmark accuracy evidence. Runtime smoke probabilities should be compared only with their own fixture, model state, and environment.

Hierarchical extension tables are preserved outputs of a separate study. Its source code specifies the expected raw-run layout. The report keeps its populations, adaptation policy, and uncertainty estimates distinct from the selected candidate's hosted feedback and local component replay.
