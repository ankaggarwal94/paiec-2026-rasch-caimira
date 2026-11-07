# Exact submit_gate implementation audit

## Artifact identity

- Archive: `submit_gate.zip`
- Size: 190,003 bytes.
- SHA256: `ddb2ad28e3887324f381777aedcefd75ec22515f09144c966f5df273c80b1a31`.
- Library file: `libfile_fde3_f-8dece3220ff88191b411c52686b950fa_af43b26cb89c819191131945df93442c`.
- Library timestamp: 2026-11-07T01:40:30.019895Z.
- Local archive: `evidence/gate/submit_gate.zip`.
- Extracted source: `evidence/gate/submit_gate/`.
- Scope: static code and serialized-state inspection. This audit launches zero training, inference-service, hosted scoring, or submission jobs.

## Implemented predictor

The entry point loads a JSON-serialized `paiec.predictors.irt_mix:IRTItemMix` instance. This is a separate hierarchical IRT predictor with learned item and subject priors, benchmark-local online adaptation, and a signed logit combination of two inference variants.

The conditional observation model, before predictive marginalization, is

\[
\operatorname{logit}p_{s,i}=\beta_b+\delta_{s,b}-d_i.
\]

The item difficulty combines a frozen cross-benchmark prior with an online content map, group effects, and an item residual:

\[
d_i=\widehat d_i+z_i^\top v_b+\sum_k a_{ik}g_{b,k}+\epsilon_i.
\]

Sources: `paiec/predictors/irt_item.py:1-17,155-167,192-213,324-379,417-457`.

The archived state uses a 64-dimensional seeded random projection and prior variance fractions 0.3 for the content map, 0.2 for item key-value groups, 0.2 for identical-content groups, and 0.3 for item-specific residuals. It enables online empirical-Bayes adjustments for map/group scales and pair-ability variance, each with pseudo-count 10. It runs 15 adaptation iterations. These choices reside in `predictor.json` and `build_info.json`; update equations appear in `irt_item.py:324-379`.

The subject model is

\[
\delta_{s,b}=x_s^\top w_b+r_{s,b},
\qquad w_b\sim\mathcal N(\overline w,S).
\]

Subject descriptors include a learned prior, standardized identity history, reasoning flag, model size, release date, reasoning effort, and missing-value flags. Revealed labels from several subjects on the same benchmark can update shared benchmark regression weights. The state sets `subject_sharing=false`, so this implementation uses benchmark-local adaptation rather than a shared online ability across different benchmarks. Sources: `paiec/predictors/irt_lr.py:1-18,27-43`, `irt_item.py:170-175,208-213,338-344`, and the serialized state.

## Implemented gate

Both members reuse one fitted model, with `lr_on=true`:

1. A variance-adjusted update, configured as `vi=true, vi_frac=0.5`.
2. A MAP-style update, configured as `vi=false`.

The first member uses a logistic-normal approximation in its likelihood updates:

\[
\mu=\sigma\left(\eta/\sqrt{1+\pi(0.5)V/8}\right).
\]

The second member uses the ordinary logistic likelihood in the update. Both final member predictions apply the stored marginalization setting. Calling the second output a plain plug-in probability would therefore lose an implementation detail. Sources: `irt_item.py:304-314,437-457`.

Let `n` denote the revealed label count for the current subject-benchmark pair. The gate bins are 0, 1, 2-3, 4-7, 8-15, and 16+. The archived weights are

\[
p_{\mathrm{gate}}=
\begin{cases}
p_{\mathrm{VI}}, & n=0,\\
\sigma\left(2.3581\,\operatorname{logit}p_{\mathrm{VI}}
-1.7053\,\operatorname{logit}p_{\mathrm{MAP}}\right), & n\ge1.
\end{cases}
\]

Member probabilities are clipped to `[1e-4,1-1e-4]` before computing logits. Every stored gate intercept is zero. The coefficients form a signed logit calibration; a convex probability-average interpretation would be incorrect. Sources: `paiec/predictors/irt_mix.py:24,50-53,70-81,90-108`; `build_info.json/config/predictor`; `predictor.json/state`.

The class supports fold-specific development weights, but this archive stores `mix_fold_w=null` and `mix_fold_c=null`. Its code comments describe offline replay fitting, while the fitting script, fitting observations, and gate-selection receipt are absent from this ZIP. Label any empirical gate improvement UNVERIFIED until those artifacts are inspected.

## Implemented acquisition

The archive contains `labeling.py` and selects `Uncertainty(radius=0.3)`. It accepts a candidate when `0.2 <= prediction <= 0.8`, subject to two budget rules:

- A depleted label budget returns false.
- If remaining candidates are at most the remaining label budget, it accepts to fill the budget.

Sources: `acquisition.json`; `paiec/acquisition/__init__.py:28-34,63-71`. The hook falls back to accepting while budget remains after an exception, and prediction errors fall back to 0.5. These fallback behaviors are implementation facts, with execution frequency unresolved.

## Training record inside this archive

`build_info.json` lists 180 benchmarks and pins data revision `54074453da7b9911be0f48e4c560be547bc84d21`. Its wider configuration names 257 requested benchmarks, excludes thresholded labels, and supplies runtime settings. The 180-entry list is the package's build record, rather than a fresh reproduction.

Stored item features use 1,024 hash coordinates plus four dense features. Lexical tokens use up to 4,000 characters, while length/style features use up to 8,000 characters; fields include text, length, and key-value metadata. Stored subject features use 256 hash dimensions plus structured descriptors. These differ from the selected 384-dimensional Rasch/CAIMIRA candidate.

## Relationship to the selected report candidate

The selected `18-s44-rasch-caimira.zip` combines Rasch and CAIMIRA at equal probability weights. It adapts within one subject-benchmark pair and uses default acquisition. The recovered gate offers a plausible complementary extension through shared item and subject effects within a benchmark.

The parent agent's conversation lookup reports a proposed combined output of 0.25 Rasch + 0.25 CAIMIRA + 0.50 hierarchical IRT, with synthetic validation only. That statement is a conversation-derived lead. This audit has established the exact gate bytes and implementation, while the combined package and its evaluation receipts remain unresolved.

No hosted score, actual replay score, or cross-candidate improvement is established by the gate archive itself. Its reference-model modules are shipped code, while the active serialized class selects IRTItemMix. Mere presence of CAIMIRA/reference modules inside the package leaves the active combination unresolved.

Precision audit: final member probabilities marginalize logit variance through sigmoid(m / sqrt(1 + pi*V/8)); the ability-minus-difficulty equation specifies the conditional response model.
