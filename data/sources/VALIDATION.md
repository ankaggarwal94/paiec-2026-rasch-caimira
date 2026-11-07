# Validation record

The release was checked locally on 2026-11-07.

```bash
python scripts/validate_register.py
python scripts/retrieve_data.py plan
python scripts/test_retrieve_data.py
```

The register check verifies 44 unique suites, 36 lineage assignments, 10,951,005 retained binary trials, 160 native-file identities, 176 projected-file identities, 40 publisher metadata entries, and the extension's 180 declared source names. All ten preserved historical reference files match their recorded byte counts and SHA256 values.

The 160 native targets also matched the historical upstream manifest during release construction. That manifest's SHA256 is preserved in `native_files.json`. The saved source bytes of all 40 publisher metadata files reproduced the Git blob IDs returned by the publisher repository. The public release retains selected fields and the source identities, keeping benchmark content outside this package.

Eight focused utility checks passed: valid bytes, missing files, same-length corruption, wrong size, unsafe relative or absolute paths, symlink escape, preservation of a conflicting existing file, reuse of a matching existing file, and an unknown benchmark selection. The valid/missing cases share one check.

The network download branch remains pending an authorized gated-data retrieval. The complete historical 44-suite projection was outside the local checks. A future reproduction should retain its download and 176-file verification receipts alongside its training logs.
