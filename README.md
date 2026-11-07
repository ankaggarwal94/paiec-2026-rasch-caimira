# Rasch–CAIMIRA for predictive evaluation

Reproduction materials for **From Overall Ability to Item-Sensitive Adaptation: Rasch–CAIMIRA for Predictive Evaluation**, by **Ankit Aggarwal** (Stanford University) and **Maharshi Gor** (University of Maryland).

The selected submission is **18-s44-rasch-caimira.zip**, Codabench **1033935**. Its supplied formative Brier ALC is **0.160822**. Across nine subject–benchmark pairs, mean Brier error decreases from **0.212061** at zero labels to **0.126671** at 31 labels, a **40.27%** reduction. All nine pairs improve at the final budget. The paper explains the combination of a shared Rasch ability update and an item-sensitive, two-coordinate CAIMIRA update.

Companion public repository: https://github.com/ankaggarwal94/paiec-2026-rasch-caimira

**Release state:** the repository contents and reviewer supplement are prepared. The public repository exists; uploading these contents and Maharshi's collaborator invitation remain the publishing steps. `publishing/PUBLISH.md` gives the exact handoff. This package preserves the original submitted predictor and separates executed checks from historical score records.

## Start here

| Goal | Location | What it reproduces |
| --- | --- | --- |
| Run the selected method | `runtime/README.md` | Exact frozen inference, all six budgets, deterministic synthetic checks |
| Recompute reported scores | `paper/scripts/analyze_feedback.py` | Every hosted aggregate from the supplied rounded scoring tables |
| Regenerate paper figures | `paper/README.md` | Four quantitative figures and the complete pdfLaTeX manuscript |
| Fit the recovered model implementation | `development/README.md` | Fitting and learned-state export from normalized, explicitly supplied data |
| Inspect source data and rights | `data/sources/DATA_SOURCES.csv` and `DATA_RIGHTS.md` | All 44 CAIMIRA sources, versions, upstream links, and retrieval scope |
| Audit training provenance | `scripts/verify_provenance.py` | Exact CAIMIRA state linkage, source-table identities, and known replay overlap |
| Inspect the hierarchical extension | `extensions/README.md` | The separately identified gate package and preserved development tables |
| Check official release requirements | `docs/COMPLIANCE.md` | Completed checks, publication steps, and remaining recovery gaps |

## Exact candidate

The original 55,644-byte archive is `runtime/original_submission/18-s44-rasch-caimira.zip`. Its SHA-256 is:

```text
23546eb90c7469fe7ea2704879775885f1b32e3277b4b7b3a1015aa1c7c6297b
```

The files under `runtime/candidate/` are byte-identical to its three members. The original ZIP is the selected Codabench executable. The larger reviewer supplement adds documentation, development code, evidence, and report sources around that unchanged archive.

## Run inference in a fresh CPU environment

These commands target Linux x86_64 and Python 3.12. They work from the repository root in zsh or bash:

```sh
python3.12 -m venv .venv-inference
source .venv-inference/bin/activate
python -m pip --isolated install -r runtime/requirements-validated.txt
python -m pip check
python runtime/predict_cli.py --input runtime/fixtures/budgets.jsonl
python runtime/smoke_test.py
```

The recorded clean environment used Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, and PyTorch 2.7.1+cpu. Dependency installation needs package-index access. Inference runs locally with the bundled state; API credentials, a GPU, and a text-embedding service are unnecessary.

Each input line is a JSON object containing `input`, optional `labeled`, and optional `id`. Each output line contains a native numeric probability and the supplied identifier. `runtime/README.md` gives the full input contract. The fixture is synthetic and tests execution; its outputs carry zero empirical accuracy evidence.

## Reproduce the paper's calculations and PDF

```sh
python3.12 -m venv .venv-report
source .venv-report/bin/activate
python -m pip install -r paper/requirements-figures.txt
python paper/scripts/analyze_feedback.py
python paper/scripts/analyze_local_replay.py
python paper/scripts/make_figures.py
python scripts/verify_provenance.py
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error paper/main.tex
```

For Overleaf, upload the contents of `paper/`, select `main.tex`, and choose pdfLaTeX. The source includes the NeurIPS 2026 style, vector figures, generated tables, and inline bibliography. Python and network access are unnecessary for the TeX build. The main text occupies eight pages; references and appendices follow.

The scoring script reconstructs the supplied **rounded summaries**. Reproducing the organizer's hidden evaluation requires its private inputs, outcomes, and acquisition sequence. Those inputs remain with the organizer.

## Development and training

`development/` preserves the recovered historical fitting source and supplies a tested portable runner for normalized JSON Lines. Its synthetic demonstration fits Rasch and CAIMIRA with the native training implementation, exports learned tensors, and checks a fresh load. Use its separate pinned environment: the historical training helper and frozen inference package have distinct PyTorch requirements.

The exact CAIMIRA state is linked to a completed 44-benchmark fit with 9,479,991 input rows, 10,951,005 response trials, 1,345 subject–benchmark pairs, seed 17, and 300 Adam updates. `data/provenance/` preserves the receipts and freeze. The projected-table inventory records all 176 expected SHA-256 values.

Three source benchmarks also appear in the archived 153-pair component replay: `finegrain_t2i`, `medagentboard`, and `tau2_bench`, covering 25 pairs. Matching native table hashes establish shared source data. We present that replay as a component diagnostic. Exact evaluated-row overlap and results for the remaining 128 pairs need the full panel and raw result CSV.

The exact historical Rasch fit receipt, complete materialized corpus, and several native extraction scripts remain recovery targets. A fresh end-to-end reconstruction of both submitted state files is therefore **UNVERIFIED**. The release makes the currently executable paths explicit and supplies precise manifests for recovering the remaining inputs.

## Data and rights

Hosted feedback contains anonymous aggregate scores. The package includes those tables, local summary records, model states, source manifests, synthetic fixtures, and the code used in this revision. Raw benchmark material follows its publisher's access and license terms. `data/sources/DATA_RIGHTS.md` explains source-specific retrieval and redistribution status; a repository-wide data license would overstate the available rights.

The hierarchical extension retains a separate identity and training inventory. The proposed 25/25/50 Rasch/CAIMIRA/gate combination remains an experiment to evaluate.

## Verify this release

```sh
python3 scripts/verify_release.py
python3 scripts/verify_provenance.py
```

`MANIFEST.json` records the prepared file identities. Generated local outputs and virtual environments stay outside that manifest. Historical sources and the exact candidate retain their original bytes, including their original vocabulary and license notices. New paper prose follows the authors' requested wording conventions.

## Citation and assistance

Use the title and author order above, with Codabench submission 1033935 and the precise release commit used. `CITATION.cff` supplies machine-readable metadata. ChatGPT assisted with inspection, calculations, figures, exposition, verification, and reproduction tooling. Claude Code assisted with development, experiments, and the earlier hierarchical manuscript. The authors are responsible for the final claims and submission.
