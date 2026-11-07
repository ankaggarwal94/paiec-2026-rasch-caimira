# PAIEC 2026 report: Rasch–CAIMIRA

The selected method is **18-s44-rasch-caimira.zip**, Codabench **1033935**, with supplied formative Brier ALC **0.160822**. The paper explains its global and item-sensitive adaptation, complete learning curve, and a separate hierarchical extension. Author order is **Ankit Aggarwal, Stanford University**, then **Maharshi Gor, University of Maryland**.

This revision starts from commit **8a156a203deb0fe3b2ab3e9e333803feb61c9501** and preserves the author's affiliation and diagram-arrow edits. It adds recovered training provenance, complete CAIMIRA source disclosure, fresh execution evidence, and reproduction instructions. The manuscript has **eight main-text pages**, followed by references and appendices.

## Companion reproduction repository

Prepared target: https://github.com/ankaggarwal94/paiec-2026-rasch-caimira

The public repository exists. Its contents upload and collaborator invitation remain the publishing steps. The manuscript cites the public companion package for project evidence. The separate release archive contains the exact candidate, score tables, training source, source registry, tested commands, and a publishing helper that invites `maharshi95` with collaborator push access.

## Compile in Overleaf

Upload the source bundle, select `main.tex`, and use **pdfLaTeX**. Required section files, the NeurIPS 2026 workshop style, vector figures, generated tables, and inline bibliography are included. The build uses standard TeX packages. Python, network access, shell escape, BibTeX, and Biber are unnecessary for compilation.

This source bundle is self-contained. Upload it into an Overleaf project and compile with the settings above. The private editing workspace has its own synchronization state.

## Local build and score reconstruction

```sh
python3.12 -m venv .venv-report
source .venv-report/bin/activate
python -m pip install -r requirements-figures.txt
python scripts/analyze_feedback.py
python scripts/analyze_local_replay.py
python scripts/make_figures.py
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error main.tex
```

`analyze_feedback.py` contains the supplied six-budget scoring transcription and writes the two CSV tables and descriptive metrics. It preserves equal pair weighting and the distinction between unique inputs and repeated response rows. `analyze_local_replay.py` reconciles the three archived component curves and their receipts. The figure script regenerates four vector charts.

## What this revision verifies

- Exact candidate identity: SHA-256 `23546eb90c7469fe7ea2704879775885f1b32e3277b4b7b3a1015aa1c7c6297b`.
- The hosted score and all main numerical results reconstructed from the supplied rounded tables.
- Frozen candidate inference in two CPU environments, including a fresh isolated installation, at all six budgets, with support isolation, input guards, and exact process-to-process agreement on synthetic fixtures.
- A completed CAIMIRA fit linked by exact state identity to 44 benchmarks, 36 source lineages, 176 projected tables, and 10,951,005 response trials.
- Fifteen recovered historical source files and a portable native training runner, tested on explicitly synthetic data.
- Known shared source data between CAIMIRA training and three local replay benchmarks, covering 25 of its 153 pairs.
- Eight main-text pages, resolved citations and cross-references, embedded fonts, and the requested prose vocabulary.

The exact historical Rasch fit receipt, complete projected corpus, and full native extraction path remain recovery targets. Full end-to-end regeneration of both submitted state files remains **UNVERIFIED**. The public code/data release and OpenReview replacement have their own completion steps; `audits/release_requirements.md` records their scope.

## Preserved hierarchical study

Appendices F and G retain the shared manuscript's development tables and original figure from commit `23a5fc8007c42950d3f0c96b7c0980b54ef8883f`. `scripts/prepare_extension_tables.py` derives the compact appendix tables from those stored outputs. The inherited `make_numbers.py` expects the earlier research checkout's `../runs/` for full statistical regeneration. Those raw run files remain outside this checkout.

The three evidence populations remain distinct: hosted formative feedback, the archived 153-pair component replay, and the separate hierarchical study. The proposed 25/25/50 Rasch/CAIMIRA/gate composition remains an experiment to evaluate.

`SOURCE_LEDGER.md` maps the current claims to artifacts. `REVISION_NOTES.md` preserves the earlier integration history and records this revision. `VALIDATION.json` gives the current compilation and artifact checks. Earlier static audits under `audits/` retain their historical scope; this revision supersedes their statements about unavailable fresh inference and wholly unresolved CAIMIRA training provenance.
