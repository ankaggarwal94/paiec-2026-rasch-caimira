# Portable structural development reproduction

This directory supplies an executable development path for the Rasch and rank-2 CAIMIRA components. The new `develop.py` adapter validates normalized JSONL trials and calls the recovered `research.broad_transfer.predictors.fit_predictor` directly. Fifteen historical source files are preserved byte for byte under `historical_source/`; `SOURCE_MANIFEST.json` records their verified identities.

The portable path fits new coefficients to supplied data. Exact candidate inference and its purchased-label policies belong to the separate candidate runtime package. The completed historical CAIMIRA receipt provides a provenance link to the submitted component; rebuilding that historical state additionally requires the complete admitted corpus, projection rules, row order, and original runtime.

## Install and check

The validated execution uses Python 3.12.14 on Linux x86_64, CPU PyTorch 2.10.0, NumPy 2.3.5, and SciPy 1.17.0. Create a fresh environment from this directory:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-validated.txt
.venv/bin/python develop.py verify-source
.venv/bin/python smoke_test.py
```

The requirements select the official PyTorch CPU wheel index alongside PyPI. CUDA and model-download credentials are unnecessary for these commands. The historical numerical helper uses the PyTorch 2.10 precision-control API. The separate inference package also supports its own validated Torch 2.7.1 environment.

The smoke fixture contains 48 synthetic trials, 24 distinct input identities, three subject strings, and six benchmark-subject pairs. It exercises both native 300-update fits and repeats each fit with seed 17. It checks equal pair weighting, changes from initialized coefficients, exact same-seed state equality, JSON/checkpoint prediction agreement, fresh-process prediction agreement, finite unit probabilities, and rejection of malformed records. Its outputs establish functional execution and serialization behavior. They carry zero benchmark-accuracy evidence.

## Normalized JSONL contract

Each training line has exactly six fields:

```json
{"benchmark":"suite-a","subject":"{\"name\":\"system-a\"}","item":"item-001","group":"group-001","text":"Complete visible item text.","label":1}
```

`benchmark`, `subject`, `item`, and `group` are nonempty strings. `text` is a string and may be empty. `label` is a JSON integer equal to 0 or 1; Boolean labels are rejected. The `subject` field carries the exact previously serialized visible subject metadata. The adapter preserves supplied subject and item-text strings verbatim. It performs zero metadata serialization, feature-field selection, dataset downloading, split admission, or source projection.

A repeated `(benchmark, subject, item)` identity must retain identical text and group fields. Each repeated line remains a separate observed trial with its own binary outcome. Repeats are consolidated into binomial sufficient statistics by the historical `Training` class; benchmark-subject pairs receive equal total weight. Input order determines first-appearance feature indexing, so preserve order when repeating an experiment.

Prediction JSONL uses exactly the same five string fields and omits `label`. Predictions retain input-line order, including repeated query lines. The prediction command supplies zero purchased labels to the fitted component. Selected-candidate blend behavior and its online updates remain in the exact inference runtime.

## Fit and export

```sh
.venv/bin/python develop.py fit \
  --input fixtures/synthetic-training.jsonl \
  --method both --seed 17 --threads 1 \
  --output runs/toy

.venv/bin/python develop.py predict \
  --state runs/toy/caimira-state.json \
  --input fixtures/synthetic-queries.jsonl > runs/toy/caimira-predictions.jsonl
```

Replace the fixture path with validated normalized source trials for development use. The default cap is 100,000 records and can be raised with `--max-records`. Data and feature matrices reside in memory. The historical fitter has a cooperative 1,200-second deadline for each fit; large jobs also require an external execution limit and an appropriate compute plan. This adapter provides a transparent entry point; full-corpus throughput remains a separate engineering task.

The output directory must be fresh. Each method produces a learned structural JSON state, a native predictor checkpoint, and a receipt. The receipt records the input hash, adapter hash, source-manifest hash, runtime, seed, data counts, fitting diagnostics, and exported-file hashes. JSON export comes directly from the trained tensor dictionary in the historical `_structured` function. Training-only subject nuisance coefficients are excluded from the deployment state. Interrupted runs may leave a partial output directory; use its per-method receipts to identify completed fits.

## Native recipe

Both methods use stateless, signed lexical hashing with width 384 for item text and serialized subject metadata. Token counts use the historical Unicode tokenization and BLAKE2b mapping, followed by L2 normalization and float32 rounding. Rasch has rank 1; CAIMIRA has rank 2, centered difficulty heads, and learned relevance weights. Metadata features supply an inductive subject prior. Training adds sum-zero subject nuisance effects using a Helmert basis.

The historical fitter uses seed 17 by default, float32 arithmetic, and 300 full-batch Adam updates. Adam uses learning rate 0.01, betas `(0.9, 0.999)`, epsilon `1e-8`, and zero optimizer weight decay. It disables fused and foreach updates. Parameters begin as seeded CPU float64 Gaussian draws scaled by 0.01 and are cast to float32; the intercept and CAIMIRA relevance bias start at zero. Deterministic algorithms and IEEE float32 backend controls are enabled. The adapter fixes CPU execution and defaults to one Torch thread.

The objective is a pair-balanced Bernoulli loss plus three explicit penalties: a coefficient penalty of `0.5 * 0.01 * sum(parameter²)` excluding nuisance and floor parameters; `0.5 * mean(subject_nuisance²)`; and, for CAIMIRA, `1e-4 * (mean(abs(centered_difficulty)) + mean(abs(ability + intercept)))`. Repeated binary outcomes contribute their original multiplicities. Native convergence diagnostics are preserved in each receipt, including `BOUNDED_UNCONVERGED` when the fixed update budget ends before the source thresholds are satisfied.

## Historical scope

`HISTORICAL_RUN.json` records the recovered CAIMIRA lineage. The executed receipt describes seed 17, 300 Adam updates, 44 benchmarks, 36 source lineages, 176 projected tables, 9,479,991 input rows, 10,951,005 trials, and 1,345 benchmark-subject pairs. That run used CUDA with Torch 2.10.0+cu128, Python 3.13.18, NumPy 2.5.2, and SciPy 1.18.1. Its state identity links the selected CAIMIRA component to the recorded completed fit. Its earlier package used an offset adaptation policy; the final candidate applies the native CAIMIRA policy to that same frozen state.

This bundle supplies the exact fitting source and a tested portable interface. Source-table retrieval, admission and projection, exact historical row construction, full-corpus resource planning, and exact Rasch-state training lineage require their separate provenance records. CPU development fits can yield different coefficients from the historical CUDA run. The source manifest and synthetic checks provide a precise reproducibility boundary without implying equality to the submitted state.
