# Exact recovery targets

The prepared release runs the submitted frozen predictor, reconstructs reported score summaries, verifies source identities, and fits the recovered model implementation on explicit normalized data. These additional records would close the remaining historical reconstruction gaps.

## Rasch fit and selected composition

The November 3 slate archive's manifest has SHA-256:

```text
174c0410a1ca7e89e57b086a04398c717206358c869d2313630c30e69e37dea0
```

It locates the following objects in `packs/chunks-001.bin`, whose full size is **90,240,994 bytes** and SHA-256 is:

```text
570840840b6ea3f4a1b87c3c56632a6c7a67ee90f8133dbdc49c3618a41345e5
```

| Restored object | Inclusive packed-byte range | Encoding | Restored bytes | Restored SHA-256 |
| --- | --- | --- | ---: | --- |
| `structured/fits/rasch.json` | 84067153–84067741 | zlib | 1,004 | `ad845d27808a1463d07b36de51dbad3dcba9cbdec6f7e95aedc2c8d749e102ac` |
| `structured/fits/rasch.pt` | 84067742–84078722 | zlib | 21,125 | `cace16dede70557eeba9d481040e05fd8bdb43998bc169f6a546c7df4c759873` |
| `structured/receipts/s44-rasch-caimira.json` | 86201628–86203227 | zlib | 3,158 | `54c3cc0d5ff9ba05068fa8eb4863430d596aac72f0f82a6c342be4d1d1d5eedf` |

Verify the full pack hash, decompress the listed ranges, and verify each restored object's byte count and hash. Inspect the fit receipt's source, seed, runtime, and checkpoint linkage before extending the exact historical Rasch claims. The pack and restored checkpoint remain outside this release until their bytes are recovered and reviewed.

The matching original tools are also identified in the archived manifest:

- `structured/tools/fit_models.py`: 3,567 bytes, SHA-256 `ede843949794ee8be743dd703a79b72cdaa6336e8432f9d9a66a015fb7462b95`.
- `structured/tools/build_candidates.py`: 6,131 bytes, SHA-256 `073890e17e27daae15edd6786b8dd8af63c1c6976b5d87b04075498edf57003c`.
- `structured/tools/score_candidates.py` and the detailed composition replay records.

## Full source projection

The historical `analysis-source.zip` is 1,968,531 bytes. It is a recovery lead for the missing benchmark extractors. Its content needs inspection and identity checks against the frozen worker inventory. The large original worker archive also contains the materialized corpus; its SHA-256 is `9bb21a2a68d8a191672043ff9072e1ddde8d3735ef2582d8dba97e4bea1a21e5`.

The recovered source in `development/historical_source/` supplies the fitter and its dependencies. The admitted 44-suite projection requires the remaining catalog/native extractors and exact row construction. Follow `data/sources/RETRIEVAL.md`, then require a successful check of all 176 projected files before attempting historical state reproduction. Source permissions govern any release of those data.

## Local replay and hierarchical study

The archived local `results.csv` is 6,781,861 bytes. Its recorded Dropbox content hash is `e4ed942c6bfe18d07c2c676c38e455f1eb9706d82e353fc113c2d18ca7a37afb`; this is Dropbox's block-based hash scheme, distinct from an ordinary whole-file SHA-256. Recover it together with the frozen panel identified in Appendix C to calculate the 128-pair sensitivity analysis and inspect exact evaluated-row intersections.

The hierarchical study's raw `runs/` records are specified by `paper/make_numbers.py`. Recover those exact records to regenerate uncertainty intervals and its original figure from responses. Current tables and the figure preserve the earlier committed outputs.

## Retrieval outcome during preparation

Direct archive transfers ended during verified-TLS handshakes, before content transfer. The available text extraction path has a size limit below the full local result CSV, and the alternative document service requires an unavailable account license. The prepared package records these bounded retrieval gaps without substituting adjacent experiments or inferred outcomes.
