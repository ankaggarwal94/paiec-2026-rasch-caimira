# Integration of the Rasch–CAIMIRA submission report

## Main report

- Ankit Aggarwal is first author; Maharshi Gor is second, with his existing University of Maryland affiliation preserved.
- The report centers `18-s44-rasch-caimira.zip`, submission 1033935, with exact archive identity and supplied Brier ALC 0.160822.
- The title and introduction emphasize the conceptual advance: overall ability correction plus item-sensitive adaptation.
- Eight main-text pages explain the implementation, update geometry, all six label budgets, pair-level results, calibration, component evidence, and the closing extension.
- The main results lead with the 40.27% endpoint Brier reduction, improvement in every evaluated pair, and substantial gains after seven labels.
- The exact shared-shift/content-contrast identity clarifies the rank-two model. The final ensemble averages branch probabilities.

## Integration with Maharshi's manuscript

The complete shared repository was read at commit `23a5fc8007c42950d3f0c96b7c0980b54ef8883f`, and every fetched file matched its Git blob hash. The hierarchical IRT model in that manuscript belongs to the same implementation family as `submit_gate.zip`.

The closing section preserves shared item effects, benchmark-local subject regression, variance-aware inference, and signed VI/MAP calibration. It defines the proposed 25/25/50 composition as a future matched experiment. The 160-benchmark evaluation fit and the gate archive's 180-benchmark build retain separate provenance.

Appendices F and G preserve the original generated results, layer ablations, and figure. The earlier external Rasch/rank-2 comparison is explicitly attributed to the manuscript, with its unresolved archive identity, different acquisition policies, and reported devbench descriptor-collision exclusion. The exact-hash 153-pair component replay remains the selected candidate's local comparison.

The original generation script remains available for a parent checkout containing `runs/`. A small new script selects the compact appendix columns from its committed tables. The earlier complete manuscript and bibliography remain available in Git history.

## Synchronization and remaining evidence

GitHub access is available for the intended repository. The report source is prepared for its existing Overleaf integration. Browser access failed for the live project, leaving its pending edits and synchronization state unresolved. Pull the shared repository through Overleaf after reviewing those edits.

The selected candidate's exact training recipe and complete overlap manifest remain UNVERIFIED. Raw hierarchical study runs require access to the parent research repository. These specific gaps are consolidated in the evidence section and appendices while the main text focuses on the supported conceptual and empirical results.


## Reproduction and public-release revision

Preserved commit 8a156a203deb0fe3b2ab3e9e333803feb61c9501, including Stanford affiliation and diagram arrows. Added state-linked CAIMIRA training provenance, all 44 source names, source-specific data access, fresh CPU inference evidence, and the portable training runner. Corrected the local replay scope after verifying three shared training-source benchmarks covering 25 pairs. Added the prepared public repository URL and retained explicit publication status. The exact candidate and original hierarchical tables/figure remain unchanged.

## Public-reference cleanup

Project evidence now cites the public companion reproducibility package. The unpublished hierarchical draft is described only as an earlier internal study, with generated evidence included in the companion materials. Twelve publisher regression tests and four local publication-flow tests cover the empty-repository repair.
