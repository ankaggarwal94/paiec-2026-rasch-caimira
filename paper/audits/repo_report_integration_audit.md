# Repository integration review

## Reviewed scope

Read the complete original manuscript, bibliography, report-generation script, README, and committed generated tables under `work/paiec_repo_base`. Compared those sources with the integrated `work/paiec_repo/sections/page08.tex`, `appendix_shared.tex`, `references.tex`, `main.tex`, and the two compact extension tables. The original generated figure was checked by SHA-256.

This review is read-only for report files. It checks scientific description, empirical attribution, numerical preservation, author metadata, candidate identity, and the requested lexical constraints. Fresh model execution and regeneration from raw research runs remain outside this review.

## Outcome

The integration preserves candidate 18 as the main evaluated submission and presents the hierarchical IRT study as a complementary research line. The original study and the archived gate share an architectural lineage. Their fitted states and evaluation populations remain separately identified.

One wording correction was sent to the parent editor: in `appendix_shared.tex`, replace “stronger initial Brier error” with “lower initial Brier error” and “stronger endpoint Brier error” with “lower endpoint Brier error”. These replacements preserve the intended numerical comparison and make the preferred direction explicit.

## Verified integration properties

- Main author order and PDF metadata are Ankit Aggarwal followed by Maharshi Gor. University of Maryland is attached to Maharshi's author block.
- The selected candidate's formative ALC remains 0.160822. The hierarchical study's 0.2077 result is explicitly assigned to a separate 20-benchmark protocol.
- The shared study's 160-benchmark hold-out fit is distinguished from the gate archive's 180-benchmark build record.
- The hierarchical model and signed gate match the original manuscript's method description and the existing gate implementation audit.
- The proposed 25/25/50 probability composition remains an experiment to evaluate.
- Every selected value in `generated/tab_extension.tex` matches the corresponding committed original main-table column.
- Every selected value in `generated/tab_extension_ablation.tex` matches the corresponding original ablation-table column.
- The table caption distinguishes stored aggregate ALC metrics from benchmark-averaged paired contrasts and their 95 percent normal intervals.
- The appendix preserves the original study's 20-benchmark and 75 percent pair-seed improvement counts as attributed historical results.
- The earlier external Rasch-plus-rank-two comparison remains separate from exact candidate identity. Its archive hash linkage is explicitly UNVERIFIED.
- The earlier external comparison's random acquisition is distinguished from Gate's radius-0.30 acquisition.
- The external replay's duplicate-descriptor failure is attributed to the source report; the selected candidate's matching logic and 31-label guard are independently supported by its code.
- The optional LLM-prior row remains a separate research variant.
- Four hold-out looks and the earlier development split for gate fitting are preserved as study context.
- The two added appendix tables and all four scoped manuscript files passed the lexical scan for the prohibited token, em dash, and TeX triple-hyphen construction.

## Preserved figure

Both original and integrated `generated/fig_results.pdf` have SHA-256:

`9cf4d3c95f643d7e3bb38bb974d8555d2ad2396896fe51de749ff413f585fc00`

## Evidence boundaries

The original `make_numbers.py` reads `pairs.csv` and `metrics.json` from run directories in the parent research checkout. The inspected report snapshot supplies committed outputs and generation code. The integrated text accurately presents those outputs as preserved historical results and leaves full regeneration as a separate evidence step.

The original script computes paired differences by subject, benchmark, and seed, then averages within benchmarks for its reported benchmark contrasts. Its normal intervals use the benchmark-level sample standard deviation and the factor 1.96. The integrated appendix preserves this distinction, avoiding arithmetic comparisons between differently weighted summaries.

Static CaimiraLite remains labeled as a static reference model. It stays distinct from the adaptive rank-two CAIMIRA branch inside the selected candidate.

## Editor resolution

The integration adopts the requested precision edits: stored aggregate metrics in the caption; 180 benchmarks in the build inventory; lower Brier scores for comparisons; and frozen ability summaries in the subject regression. README and the source ledger now describe the completed manuscript integration and the remaining Overleaf connection gap.
