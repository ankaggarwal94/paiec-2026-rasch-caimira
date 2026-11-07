# Offline fitting and serialization scope

## Exact target

This inspection concerns the three-file candidate archive with SHA-256 `23546eb90c7469fe7ea2704879775885f1b32e3277b4b7b3a1015aa1c7c6297b`. It is separate from the wider project's recovered training source and historical receipts.

## Available computation

The candidate loads two frozen JSON states and constructs CPU inference objects at `model.py:533-535` and `1082-1084`. Each state contains `schema`, `config`, `parameters`, `centers`, and `orientation`. The public entry point at lines 1103-1109 averages the native CAIMIRA and Rasch-offset predictions.

The active CAIMIRA native update calls SciPy at line 139. Its objective is logistic likelihood plus a unit Gaussian penalty around the subject prior. The second factory carries an identical helper at line 688, while its selected Rasch-offset path uses 42 scalar bisections at lines 621-632. Both active updates leave the fitted item and metadata maps fixed.

Whole-source inspection and the AST inventory found zero offline fit or train entry points, Torch optimizer construction, backward calls, optimizer steps, or training-data loaders. The generic parameter-shape helper includes a nuisance shape option, but the selected inference constructor accepts only the frozen parameter shapes.

## What state export preserves

`ResidentStructured.from_state` loads a frozen state at lines 352-354 and 901-903. Its constructor copies the imported dictionary into `_state` at lines 350 and 899. `export` at lines 356-357 and 905-906 returns a JSON deep copy of that saved dictionary.

Consequently, export is a round trip of the imported inference state. A separately written fitter that mutated resident tensors would still need its own checkpoint serialization, because this export method reads the saved constructor dictionary. Forward computations at lines 359-386 and their second-factory equivalents also use Torch inference mode.

The archive therefore supplies sufficient state for inference, but its export method is insufficient as an offline training checkpoint writer. The stored ridge and regularization fields describe configuration; they alone leave the data order, optimizer schedule, initialization, centering procedure, and checkpoint selection unspecified.

## Historical reproduction boundary

A new optimizer built from the forward equations would be a new implementation until matched to the historical worker and its exact inputs. The wider provenance audit is recovering that training source, data freeze, and receipts. This runtime package makes zero claim that a new fitting process regenerated the submitted coefficients.

Frozen-state inference has now passed the clean-environment execution suite. Historical offline fitting remains a separate evidence path, evaluated against the recovered training artifacts.
