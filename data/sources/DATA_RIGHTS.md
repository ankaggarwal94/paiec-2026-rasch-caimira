# Data sources, terms, and release decisions

## Release scope

This release provides the source register, historical receipts, file identities, and retrieval utilities. It carries zero raw benchmark rows, model response transcripts, images, audio recordings, clinical records, or other benchmark content. Each reader obtains source data through the applicable provider and preserves its conditions.

The 44 rows of [DATA_SOURCES.csv](DATA_SOURCES.csv) identify every benchmark in the selected training freeze. Source membership follows the executed freeze and its receipts. Organizer inventory membership supplies the recorded competition eligibility lineage; it supplies no additional copyright grant. The recorded inventory export has SHA256 `882809d047c6337e265e59d14250cefd1fc0300e5a8e9f7c04bfda05c960cd5a` and is linked in every row.

## MeasurementDB access and attribution

The [MeasurementDB dataset card](https://huggingface.co/datasets/aims-foundations/measurement-db) requires access approval and acceptance of its conditions. Its CC BY-SA 4.0 statement covers AIMS's original curation contributions where AIMS owns rights, including selection, organization, schema, metadata, and normalization. Each underlying benchmark and third-party asset retains its own upstream licenses and terms. The dataset card describes version-pinned tables and per-benchmark metadata as the source of that information.

The attributed metadata extract in `references/publisher-source-metadata.json` derives from **AIMS Foundation, MeasurementDB**, at [code commit 287495967fc276275c3b466dfede2cc3534e830c](https://github.com/aims-foundations/measurement-db/commit/287495967fc276275c3b466dfede2cc3534e830c). The extraction selects source URLs, revisions, and license fields, adds historical receipt links, and leaves benchmark content with its original providers. Preserve this attribution and [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) for the covered curation contributions. Any separate repository software license applies only within its stated scope.

The 31 historical catalog license fields match the corresponding publisher metadata fields exactly. For the nine historical native admissions, publisher metadata supplies the recorded license field at the stated code checkpoint; historical native file hashes provide a separate identity check. The register preserves license values such as `unknown` and conflicting declarations. These values identify evidence boundaries for the next redistribution review.

## Source-specific decisions

The release decision for every suite is **metadata and retrieval recipe only**. The following issues determine the additional review before a raw-data redistribution:

| Sources | Recorded condition or gap | Required next step |
| --- | --- | --- |
| ClashEval, REAL, ResearchCodeBench, MedHELM MedCalc | The available license field is `unknown`. | Obtain an applicable grant or documented basis for the proposed content and use. |
| MIR | Pinned GitHub code has an Apache License 2.0 header. Dataset and response-release terms remain unresolved. | Establish terms for the task data and released GPT-4o outputs separately from the code. |
| FineGRAIN T2I | Dataset README/card declare MIT; separate Croissant metadata declares CC-BY-4.0. | Preserve both statements and resolve their respective scope. |
| AtmosSci-Bench, DIS-CO, MathArena, WorldCentralBanks | Recorded terms include noncommercial conditions or a mixture of source licenses. | Review the intended use and each incorporated component. |
| WorkArena | AgentRewardBench custom dataset terms cover the released trajectories. | Apply those terms and third-party conditions to derivatives. |
| Visual Riddles | Upstream card records a commercial-training restriction and retained image rights. | Review the card and image rights for the proposed use. |
| MedAgentBoard | Admitted data comprise PubMedQA, MedQA, and VQA-RAD. The broader archive includes controlled clinical content outside this slice. | Preserve the admitted subset and its component terms; exclude the controlled records. |
| GoodAI LTM | Repository MIT declaration coexists with component terms and GPT-generated-test restrictions. | Review the three admitted task components and upstream usage conditions. |
| SWE-bench Pro | MIT covers the Scale AI harness; original task repositories include GPL, AGPL, and Apache declarations. | Retain original project terms and attribution for the actual task content. |
| Other suites | The register records a declared license, frequently with component-specific qualifications. | Preserve notices and attribution and review third-party assets before redistribution. |

These are source statements and conditional release decisions. Each row includes its evidence pointer and scope. Public availability and organizer inclusion serve different purposes from source-specific redistribution permission.

## Supplemental evidence

AndroidWorld's pinned provider license has Git blob `654aa773711a7915627d351266ff5ad3f2a9bb04`, matching its historical source receipt. CYBENCH's historical receipt records `Apache-2.0` and the license file SHA256. SWE-bench Pro's receipt preserves representative original project license URLs and revisions. MIR's source receipt identifies the data and grader revisions; its code license evidence is recorded separately in `references/supplemental-license-evidence.json`.

The four source receipts preserve their original bytes. Their original wording serves as historical evidence. Newly written release prose and current retrieval status are kept in this register and the retrieval guide.

## Hierarchical extension

`EXTENSION_SOURCE_NAMES.csv` discloses all 180 benchmark names declared by the separate `submit_gate` package. The associated extract records source file SHA256 `2115c25decd1c35d7d36d93fb99bdc8a620455c96c19063ba06a2a49dc137c13`. Its declaration uses MeasurementDB revision `54074453da7b9911be0f48e4c560be547bc84d21`. The 44-suite rights review applies only to the overlap and the source scopes documented here. The other extension sources require their own terms review before raw-content redistribution.

## Evidence date

Publisher metadata and access conditions were inspected on 2026-11-07. The original training freeze and source receipts remain immutable historical records. Recheck provider access and source conditions when obtaining data for a new use.
