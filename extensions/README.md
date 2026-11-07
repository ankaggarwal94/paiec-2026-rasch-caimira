# Hierarchical IRT extension

The paper's final main section discusses the separately developed hierarchical IRT and signed-gate method. It is a possible extension to the selected Rasch–CAIMIRA submission.

`gate/submit_gate.zip` preserves the inspected 190,003-byte extension archive. Its SHA-256 is:

```text
ddb2ad28e3887324f381777aedcefd75ec22515f09144c966f5df273c80b1a31
```

The extracted `gate/` source and state preserve that archive's bytes. The package selects `paiec.predictors.irt_mix:IRTItemMix`; `build_info.json` lists 180 build benchmarks and the pinned MeasurementDB revision. The archived build configuration contains a historical local data-directory path. A new run should supply its own data location.

The earlier hierarchical study's tables and figure remain under `paper/generated/`. Its evaluation fit used 160 training benchmarks and a 20-benchmark hold-out, distinct from the 180-benchmark deployment inventory. The inherited `paper/make_numbers.py` documents the expected `../runs/` inputs for full statistical regeneration. Those raw run files remain a recovery target.

Static inspection verifies the extension's equations, state, member settings, and acquisition rule. The current revision performs zero fresh gate accuracy evaluations. The proposed 25% Rasch, 25% CAIMIRA, 50% gate mixture is unscored. Keep this extension separate from the selected Codabench candidate and its reported formative score.
