# Competition requirements and release status

This audit concerns the exact selected candidate **18-s44-rasch-caimira.zip**, Codabench **1033935**, and the revised paper based on report commit `8a156a203deb0fe3b2ab3e9e333803feb61c9501`. It preserves the user's Stanford affiliation and revised diagram arrows.

The primary requirements are the [official PAIEC competition page](https://aims.stanford.edu/competition), accessed November 7, 2026 UTC, and the OpenReview submission form supplied by the author. The extended report/code deadline is **November 6, 2026, 23:59 AoE**, equivalent to **November 7, 2026, 11:59 UTC**. The Codabench platform's later competition end date is a separate date.

## Requirement matrix

| Requirement | Evidence and implemented action | Current disposition |
| --- | --- | --- |
| Approximately 4–8 main-text pages in the NeurIPS template | `paper/main.tex`, included 2026 workshop style, compiled page audit | Eight main-text pages; references and appendices follow |
| Describe data, method, implementation, setup, results, limitations, and rerun instructions | Main sections 1–8; Appendices A–H; executable reproduction guides | Included, with exact recovery gaps identified |
| Link released materials from the report | Intended repository URL in Appendices E and H and bibliography | Link prepared; public publication remains a release step |
| Identify the selected Codabench method | Exact original ZIP and three member digests; report identifies 1033935 | Verified archive identity |
| Release inference code | `runtime/candidate/`, exact original ZIP, tested CLI | Complete locally; public publication pending |
| Release development code | `development/historical_source/`, portable native fit runner, report-analysis scripts | Recovered source and tested runner included; original Rasch receipt and complete extraction pipeline remain partial |
| Disclose every training source | 44-source CAIMIRA registry, 176 projected-table digests, separate gate inventory | CAIMIRA disclosure verified; exact historical Rasch corpus linkage remains UNVERIFIED |
| Release permitted training additions beyond MeasurementDB | Native supplement manifests, upstream locations, rights registry, retrieval instructions | Partial: complete materialized supplements and some extraction scripts remain recovery targets |
| Use permitted inventory sources | Frozen corpus and publisher source records | Historical source records supplied; per-source conditions remain controlling |
| Adapt using supplied attributes and labels within the budget | Exact candidate source; tests for 0/1/3/7/15/31 labels, support isolation, and invalid-label guards | Passed fresh CPU checks |
| Restrict inference networking | Selected predictor uses local lexical features, bundled parameters, and numerical optimization | Local execution path verified |
| Preserve private evaluation data | Only anonymous scoring summaries are included | Private inputs and outcomes remain outside the release |
| Disclose coding assistance | Acknowledgments name ChatGPT and Claude Code and their roles | Included |
| Upload a PDF and supplementary ZIP of at most 100 MB | PDF and complete reviewer archive built from these files | Local artifacts prepared; OpenReview replacement is a separate step |
| Supplement includes submission code and clear reproduction instructions | Original ZIP, extracted inference, pinned environments, commands, data manifests | Included; full historical training reconstruction has the stated gaps |
| All authors have OpenReview profiles; author order and affiliations agree | Paper: Ankit Aggarwal first, Maharshi Gor second | Paper verified; current OpenReview metadata needs account-side confirmation |
| Email sharing, release-on-acceptance, and license fields | Requirements in author-supplied form | Current submitted field values remain outside this audit |
| Public repository under Ankit's account; Maharshi collaborator access | Prepared target `ankaggarwal94/paiec-2026-rasch-caimira` and publishing helper | Repository creation and invitation blocked by available management interfaces |

## Release conclusion

The paper's formatting, selected-candidate identity, numerical score reconstruction, and fresh frozen-inference checks pass. The reviewer package contains concrete runnable code and exact evidence. Full presentation-release compliance remains **PARTIAL** until public publication, recovery or resolution of the remaining development/data release items, and OpenReview replacement are completed. Successful inference alone leaves those separate obligations unchanged.

## Concrete remaining items

1. Publish the prepared repository using `publishing/PUBLISH.md`; verify public visibility and Maharshi's push permission. A pending invitation is distinct from accepted collaborator access.
2. Recover the exact Rasch fit receipt and associated original fitting/extraction records identified in `docs/RECOVERY.md`.
3. Complete each permitted native-supplement release using the source registry and its license conditions, or document organizer-accepted access arrangements.
4. Replace the OpenReview PDF and supplement, preserve selected ID **1033935**, and check title, author order, affiliations, consent fields, and the chosen license.

No OpenReview edit, Codabench resubmission, or public GitHub creation was performed by this package's preparation process.

## Publisher repair status

The public target now exists and is empty. The corrected Bash/Python publisher is bundled; artifact upload and collaborator invitation remain owner-run steps. Paper citations now point to the public companion package. Earlier data-recovery and submission-update items retain their stated scope.
