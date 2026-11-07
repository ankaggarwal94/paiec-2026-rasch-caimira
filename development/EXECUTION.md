# Executed validation

The frozen-source check passed for all 15 files. Both structural models were fitted on the supplied 48-trial synthetic fixture, using the unchanged native 300-update routine, seed 17, float32, CPU execution, and one Torch thread.

The complete smoke test passed first in a dedicated environment with inherited NumPy/SciPy, then in a fresh isolated virtual environment installed from `requirements-validated.txt`. Both environments produced identical validation summaries and learned states. The isolated NumPy, SciPy, and Torch modules all loaded from that environment. Both smoke commands exited with code 0 and empty stderr.

| Check | Rasch | CAIMIRA |
|---|---:|---:|
| Native optimizer updates | 300 | 300 |
| Native convergence status | CONVERGED | CONVERGED |
| Largest coefficient change from initialization | 0.9486150150 | 0.9500451391 |
| JSON versus native-checkpoint predictions | Exact match | Exact match |
| Fresh-process predictions | Exact match | Exact match |
| Same-seed repeated learned state | Exact match | Exact match |

These are synthetic execution checks. Objective values and forecast ranges in the machine-readable records describe the toy fixture alone.

Evidence files:

- `smoke-results/summary.json`: final functional checks and runtime versions.
- `smoke-results/*-state.json`: actual learned structural tensors serialized as JSON.
- `smoke-results/*-receipt.json`: native fit diagnostics and hashes.
- `smoke-results/*-predictions.jsonl`: fresh-process forecasts.
- `logs/smoke-stdout.json` and `logs/smoke-clean-stdout.json`: full command outputs.
- `CLEAN_ENVIRONMENT_VALIDATION.json`: isolated dependency origins, commands, and exit codes.
- `SOURCE_MANIFEST.json`: preserved historical source identities.

The smoke script creates native `.pt` checkpoints in a temporary directory, loads them with the historical `weights_only=True` loader, checks their predictions, and then removes the temporary directory. Running the documented `fit` command creates persistent JSON states, native checkpoints, and receipts at the chosen output path.

The new adapter leaves the historical source, selected submission archive, and submitted coefficients unchanged. Full-corpus reconstruction remains a separate data and provenance task.
