# Retrieve native data and verify the frozen projection

## 1. Exact targets

The 44-suite corpus contains 10,951,005 retained binary trials. `native_files.json` lists four native tables for each of 40 MeasurementDB suites at revision `54074453da7b9911be0f48e4c560be547bc84d21`. Those 160 files total 1,002,361,496 bytes, before any local cache duplication. The four remaining suites have separate recipes below.

The native download manifest is a selected extract of the historical upstream manifest with SHA256 `b2efb2d99bb9306a035a1748dc01338bf34025312423f3146d0d310404e469b9`. Every selected file matches its admission receipt independently. Projected data identities appear in `projected_files.json`; the 176 projected files total 331,651,941 bytes.

## 2. Fetch the 40 mirror suites

Obtain authorized access through the [MeasurementDB page](https://huggingface.co/datasets/aims-foundations/measurement-db), accept its conditions, and configure the Hugging Face client's local credential mechanism. Keep credentials and downloaded benchmark data outside the public release tree.

Run these commands from this directory:

```bash
python -m pip install -r requirements-data.txt
python scripts/retrieve_data.py plan
python scripts/retrieve_data.py fetch-native --root /path/to/native-snapshot --receipt /path/to/native-fetch-receipt.json
python scripts/retrieve_data.py verify-native --root /path/to/native-snapshot
```

A single-suite rehearsal uses the same immutable targets:

```bash
python scripts/retrieve_data.py fetch-native --benchmark algotune --root /path/to/native-snapshot
```

Files land under `tables/<benchmark>/formatted_tables/`. The provider call uses an explicit full revision. Each downloaded file must match the recorded byte count and SHA256 before the utility creates its final destination. Conflicting existing files cause an error and stay intact. Access failures stop retrieval; the utility has zero fallback to a newer revision or an alternate corpus.

The [official client documentation](https://huggingface.co/docs/huggingface_hub/v0.34.4/en/package_reference/file_download) describes `hf_hub_download` and its revision argument. The utility's download branch requires `huggingface_hub==0.34.4`; its planning and verification branches use only Python's standard library.

## 3. Reconstruct or supply the historical projection

The source tables require benchmark-specific admission, grouping, visible-field serialization, and missing-grade handling. The historical reconstruction record names `research.broad_transfer.catalog` for the 31 expanded catalog suites, `archives_native` for the nine historical mirror suites, and separate AndroidWorld, CYBENCH, MIR, and SWE-bench Pro extractors.

For a complete historical source checkout, the recorded catalog command is:

```bash
python -m research.broad_transfer.catalog --snapshot /path/to/native-snapshot --organizer /path/to/organizer.csv --cohort 6 --suite algotune --output /path/to/projected-corpus/catalog/algotune
```

Treat this as a documented historical command. The current metadata package supplies retrieval and hash verification; execution of the full historical projection depends on the archived extractors and their dependencies. Their availability is tracked by the main reproduction package. A downloaded native table remains a source table until the historical projection checks succeed.

Use the exact output directories in `references/combined-sources.json`. The example output above illustrates a single suite; its authoritative directory is the corresponding manifest entry. An organizer export must match SHA256 `882809d047c6337e265e59d14250cefd1fc0300e5a8e9f7c04bfda05c960cd5a` for historical reconstruction. A fresh export can differ.

Once the full projected tree is available, run:

```bash
python scripts/retrieve_data.py verify-projected --root /path/to/projected-corpus --receipt /path/to/projected-verification.json
```

A successful result requires all 176 expected files to match both size and SHA256. Missing files, changed bytes, and unsafe paths yield a failing exit status. This is the acceptance gate for claiming the historical projected corpus.

## 4. Supplemental sources

### androidworld

Use [babelcloud/android_world_benchmark](https://github.com/babelcloud/android_world_benchmark/tree/f788f4eb3cb91213b532112c9874e7d1d7eb796a) at commit `f788f4eb3cb91213b532112c9874e7d1d7eb796a`. The historical receipt names the GBOX trajectory tree `e1dd9661815effe8cbcb340707a8c411b2438609`, report blob `c7690861f7c36e89f2ac1363ba7253c297f14763`, and license blob `654aa773711a7915627d351266ff5ad3f2a9bb04`.

The [author leaderboard export](https://docs.google.com/spreadsheets/d/1cchzP9dlTZ3WXQTfYNhh3avxoLipqHN75v1Tb86uhHo/export?format=csv&gid=0) had 9,203 bytes and SHA256 `fa55b5154926e6027efa02dda6a07d15e008b3e314adbc20d0980ce537aca8f0`. A current export may differ. Preserve the 116 observed attempts, comprising 100 successes and 16 failures, and the three visible configurations. The four projected-file targets appear in `references/archives-androidworld.json`.

### cybench

Use MeasurementDB legacy revision `589ccfdb8e82e6e0b5e35e9d23cd83a6df85018f`. `references/archives-cybench.json` records four exact native files, their byte counts, and SHA256 values. Their legacy repository path mapping requires the historical extractor or source inventory. The recipe leaves that mapping explicit rather than guessing paths.

The [upstream CYBENCH repository](https://github.com/andyzorigin/cybench/tree/ca483c6443f17a20ba7d6bac3d274859a1721e51) is pinned to `ca483c6443f17a20ba7d6bac3d274859a1721e51`. Its license hash and the comparison to [paper tables 10 and 11](https://arxiv.org/html/2408.08926) are recorded in the receipt. Preserve 320 unguided and 320 subtask-guided binary outcomes. The 320 fractional outcomes remain outside this training slice.

### mir_bench

Use the [MIR-Bench task dataset](https://huggingface.co/datasets/kaiyan289/MIR-Bench) at data revision `16a95704f98021ef70daaff6cd77c38a99f7e934` and the [MIR-Bench code repository](https://github.com/KaiYan289/MIR-Bench/tree/4db07db3c9187e71e2552526172a580b0c9b5667) at grader revision `4db07db3c9187e71e2552526172a580b0c9b5667`. Follow the author's release links for original GPT-4o outputs. The receipt pins these exact files:

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `MIR-Core.parquet` | 123,869,113 | `c322fdbc83b212993ffb64cc9a6ad29fb06744679bb2f526e98ed4f2d7271cfe` |
| `evaluate.py.txt` | 4,025 | `8885a887e8915335ec3426ed4357799eebde9d72453f3b3348584429449c9312` |
| `manyshot_benchmark#64acc_quad_512.json` | 103,297,545 | `a69eafcb9b9048c4438d4f25feb3ed879abb321e89fdc353e0329d4440bc9593` |

The historical grader's behavior is part of the target. Its receipt records 515 changed outcomes under a missing-import repair, so an improved grader would define a different dataset. Confirm dataset and response-release terms before obtaining or redistributing those records.

### swe_bench_pro

Use [Scale AI's repository](https://github.com/scaleapi/SWE-bench_Pro-os/tree/66f92766bba642462d4bbe5479e83f91f9211862) at commit `66f92766bba642462d4bbe5479e83f91f9211862` and [task dataset](https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro) at revision `7ab5114912baf22bb098818e604c02fe7ad2c11f`.

The historical receipt links the official `traj/README.md` to [Docent collection 032fb63d-4992-4bfc-911d-3b7dafcb931f](https://docent.transluce.org/dashboard/032fb63d-4992-4bfc-911d-3b7dafcb931f). Its public metadata endpoint is recorded in `references/archives-swe-bench-pro.json`, alongside exact hashes for the captured metadata, IDs, task items, independent export, and license records. Current API responses require hash verification against the historical capture. Preserve the observed 9,729 outcomes rather than assigning outcomes to missing subject-item pairs.

## 5. Execution status and stopping conditions

The release construction reconciled all source and projection manifests locally and exercised the verifier's matching, mismatch, and path-containment behavior. It performed zero gated corpus downloads and zero complete 44-suite projection rebuilds in this session. Those steps need the provider access and historical extraction inputs described above. Keep the emitted receipts with any future reproduction run.
