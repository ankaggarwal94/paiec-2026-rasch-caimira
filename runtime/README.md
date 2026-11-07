# Exact candidate inference reproduction

This package executes the frozen predictor from `18-s44-rasch-caimira.zip`, Codabench submission 1033935. The original archive has SHA-256:

`23546eb90c7469fe7ea2704879775885f1b32e3277b4b7b3a1015aa1c7c6297b`

`original_submission/18-s44-rasch-caimira.zip` preserves that archive byte for byte. The three files under `candidate/` equal its original members. The CLI checks their hashes before loading the predictor.

## Validated environment

A fresh Linux x86_64 virtual environment installed every version in `requirements-validated.txt` successfully. Its dependency consistency check and inference suite passed. The reference environment used Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, and PyTorch 2.7.1+cpu, with glibc 2.39. PyTorch's CUDA build field is null. The selected module sets PyTorch to one CPU thread.

The pins also cover the runtime dependency closure. These are the versions used for the fresh release checks. Historical training and evaluation environments have separate records.

## Install and run

From the extracted package directory, using Python 3.12 on Linux x86_64:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip --isolated install -r requirements-validated.txt
python -m pip check
python predict_cli.py --input fixtures/example.jsonl
python smoke_test.py
```

The installation uses public PyPI and the official PyTorch CPU wheel index. Inference uses local computation and the supplied fitted states. API credentials, a GPU, and a pretrained text encoder are unnecessary. Dependency installation requires package-index access.

The CLI accepts one JSON object per line. Each request has `input`, optional `labeled`, and optional `id` fields. Its output has a probability and preserves the supplied identifier. To process all six fixture budgets:

```sh
python predict_cli.py --input fixtures/budgets.jsonl
```

Omit `--input` to read JSON Lines from standard input. Use `--candidate-dir PATH` to select a directory containing the exact three candidate files. Output goes to standard output. A malformed request or failed candidate check produces a diagnostic on standard error and exit code 2.

## Input contract

The submitted function is `predict(input, labeled=None)`. Its input is a two-element sequence containing a subject metadata dictionary and an item dictionary. The item uses `item_content` and `benchmark_id`. Subject metadata must be JSON-serializable.

Each revealed observation has the structure `((subject, item), label)`. Labels must be native Python integers 0 or 1. JSON numbers 0 and 1 produce that type. Boolean labels and labels outside the binary range are rejected.

The model keeps observations matching the complete subject dictionary and the same nonempty benchmark identifier. It accepts at most 31 eligible observations. An absent benchmark identifier selects the frozen zero-label forecast. If distinct source pairs reuse an identical subject dictionary and benchmark, their observations enter the same eligible prefix; keep that combined prefix within the limit.

The wrapper validates the JSON structure and returns the submitted predictor's native Python floating-point probability. The model code and fitted-state bytes remain unchanged.

## Fixture and completed checks

`fixtures/fixture.json` holds one synthetic query and 32 illustrative support observations. The extra observation exercises the guard. `fixtures/budgets.jsonl` supplies nested prefixes at the six scored budgets. These generated labels exercise the interface and support updates; they supply zero evidence about real-world prediction accuracy.

The executed suite passed:

- Native Python float output, finite and within the unit interval, at budgets 0, 1, 3, 7, 15, and 31.
- Exact repeated output within one process.
- Exact output agreement with two fresh CLI processes.
- Isolation from observations for a different subject or benchmark.
- The frozen-prediction fallback for a missing benchmark identifier.
- Rejection of 32 eligible labels, a Boolean label, and an integer outside the binary range.
- Unchanged candidate member hashes after execution.

The following values belong to this synthetic fixture and the captured environment:

| Revealed labels | Probability |
| --- | ---: |
| 0 | 0.43545016646385193 |
| 1 | 0.53220648590449537 |
| 3 | 0.52485780410298399 |
| 7 | 0.43329385364065376 |
| 15 | 0.50947111723309302 |
| 31 | 0.53215020873037677 |

The initial environment and the clean environment produced identical values at every budget. Their maximum absolute difference was zero. `results/reference_smoke_results.json` preserves the checked reference run. A new invocation writes `results/smoke_results.json` and fresh CLI output files. These execution checks leave competition accuracy estimates and fitting experiments to their separate evaluation procedures.

The existing reviewer supplement's two-call example also executed successfully. Its separate illustrative query produced 0.43542760610580444 at zero labels and 0.5321305488696426 after one label. The exact command, example-file hash, standard output, and standard error are recorded in `logs/supplement_example.json`.

## Evidence files

- `results/environment_clean.json`: exact package versions, module locations, platform, virtual-environment configuration, and PyTorch build information.
- `results/reference_smoke_results.json`: reference predictions, checks, candidate hashes, fixture hash, and fresh-process commands.
- `results/environment_comparison.json`: comparison of the initial and clean environments.
- `logs/clean_venv_setup.json`, `logs/clean_install.json`, and `logs/pip_check.json`: clean setup and dependency checks.
- `logs/clean_smoke_execution.json`: exact smoke command, exit status, standard output, and standard error.
- `logs/supplement_example.json`: execution receipt for the supplied supplement example.
- `results/archive_training_code_audit.json` and `TRAINING_SCOPE.md`: the distinction between frozen inference and offline fitting.
- `MANIFEST.json`: release-file identities, including the exact original archive.

## Scope of reproduction

This package reproduces execution of the supplied frozen predictor and records deterministic synthetic fixtures. Report-score reconstruction is supplied separately through the scoring-transcription script. A fresh hidden competition score requires the organizer's evaluation inputs, observed responses, and acquisition stream.

The candidate ZIP contains forward inference, state loading, and support-conditioned MAP updates. It supplies zero offline training loops. Reproducing the historical fitting process requires the separately audited training source and receipts. `TRAINING_SCOPE.md` gives the exact source boundary.

Virtual environments and generated bytecode are excluded from the release archive. The package contains source, frozen states, fixtures, and execution receipts.
