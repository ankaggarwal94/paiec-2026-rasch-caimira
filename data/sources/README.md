# Frozen training-data source register

This package identifies the 44 benchmark suites behind the selected CAIMIRA branch and supplies a strict retrieval and verification path. It contains metadata, checksums, source receipts, and small utility scripts. Original benchmark content remains with its providers.

The selected submission archive has SHA256 `23546eb90c7469fe7ea2704879775885f1b32e3277b4b7b3a1015aa1c7c6297b`. The recovered CAIMIRA receipt links its frozen state to the 44-suite training freeze. The Rasch state has a separate training-provenance scope described in the main reproducibility package.

## Contents

| File | Purpose |
| --- | --- |
| [DATA_SOURCES.csv](DATA_SOURCES.csv) | All 44 benchmark IDs, 36 lineage assignments, retained trial counts, organizer identity, upstream URLs and revisions, terms, access requirements, and retrieval pointers. |
| [DATA_RIGHTS.md](DATA_RIGHTS.md) | Source-specific rights register, attribution, and redistribution decisions. |
| [RETRIEVAL.md](RETRIEVAL.md) | Exact commands and separate-source recipes. |
| [native_files.json](native_files.json) | 160 native source files for 40 suites at one immutable MeasurementDB revision; 1,002,361,496 bytes in total. |
| [projected_files.json](projected_files.json) | All 176 historical projected fitting files; 331,651,941 bytes in total. |
| [references/](references/) | Exact historical freeze and receipts, plus clearly labeled extracts from public publisher metadata. |
| [EXTENSION_SOURCE_NAMES.csv](EXTENSION_SOURCE_NAMES.csv) | The separate hierarchical extension's declared list of 180 benchmarks. |
| [validation_manifest.json](validation_manifest.json) | Reconciled counts and the scope of local checks. |

## Corpus identity

The training freeze admits **10,951,005 binary trials from 44 suites and 36 source lineages**. Its source groups are 31 expanded catalog mirror suites, nine historical native mirror admissions, and four supplemental curated sources. The first 40 use MeasurementDB revision `54074453da7b9911be0f48e4c560be547bc84d21`. AndroidWorld, CYBENCH, MIR, and SWE-bench Pro have separate source receipts. CYBENCH uses an earlier MeasurementDB revision and an upstream audit.

For every one of the 160 native files, its path, size, and SHA256 agree with both the historical upstream manifest and the relevant admission receipt. Every projected file's SHA256 agrees with `references/freeze.json`, `references/combined-sources.json`, and its per-source receipt. These comparisons establish the exact targets for retrieval and reconstruction.

Native tables and projected fitting tables have different schemas and hashes. Native tables include `benchmarks`, `items`, `responses`, and `subjects`; projected tables include `facts`, `items`, `subjects`, and `ungraded`. Keep them in separate directories.

## Start here

```bash
python scripts/retrieve_data.py plan
python -m pip install -r requirements-data.txt
python scripts/retrieve_data.py fetch-native --root /path/to/native-snapshot --receipt /path/to/native-fetch-receipt.json
python scripts/retrieve_data.py verify-native --root /path/to/native-snapshot
python scripts/retrieve_data.py verify-projected --root /path/to/projected-corpus --receipt /path/to/projected-verification.json
```

Complete the provider's access process before retrieval. The utility uses the Hugging Face client's usual local credential mechanism, pins the complete historical revision, and accepts a file only after its byte count and SHA256 match. It preserves existing destination files and stops on a conflicting file. See [RETRIEVAL.md](RETRIEVAL.md) for source reconstruction and access boundaries.

## Evidence boundaries

The historical data revision and the publisher metadata checkpoint are separately recorded. Publisher source URLs and terms were read at code commit `287495967fc276275c3b466dfede2cc3534e830c`; that checkpoint supplies primary metadata evidence. Exact historical training bytes remain defined by the frozen table hashes. A current provider download or reconstruction earns historical equivalence only through those checks.

The hierarchical extension's 180 names are declarations extracted from its `build_info.json`. The extract retains the original file's SHA256 and removes the internal filesystem root. Thirty-four names overlap the selected 44-suite list. The extension has a separate fit and evaluation scope, and its full set of data rights requires a separate review.
