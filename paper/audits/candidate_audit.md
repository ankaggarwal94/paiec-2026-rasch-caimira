# Exact candidate audit

## Scope and verdict

Target: `upload/18-s44-rasch-caimira(1).zip`, SHA-256 `23546eb90c7469fe7ea2704879775885f1b32e3277b4b7b3a1015aa1c7c6297b`.

The static audit supports the method equations and parameter counts on manuscript pages 3 and 4. Two wording repairs make page 4 match the implementation precisely. Actual inference execution remains UNVERIFIED because the available Python environment lacks torch. Runtime smoke tests, training reproduction, and leaderboard reproduction remain outside the completed evidence.

## Manuscript repairs

1. Replace “The evaluator first filters revealed evidence” with “The exported predictor first filters revealed evidence.” The exported `predict` function performs the subject and benchmark matching at `model.py:1103–1107`.
2. Replace “Its convergence checks cover finite objectives and residual gradients” with “The solver rejects nonfinite objectives and failed solves whose largest gradient component exceeds 10^{-6}.” The exact guard at `model.py:140–141` admits a successful optimizer result irrespective of gradient magnitude and admits a failed optimizer result when its gradient residual is at most 10^{-6}.

## Completed checks

- Safely extracted the exact archive and recorded file sizes and SHA-256 hashes in `candidate_manifest.json`.
- Read the complete first model factory and inspected the complete second factory through a unified diff. The factories differ in name, selected state filename, selected family, and selected update strategy.
- Used AST inventory to establish the exported interface and factory boundaries.
- Parsed both JSON states and checked configuration, parameter shapes, finite values, centering vectors, and identity orientation.
- Traced the active interface, evidence filter, feature path, frozen prediction path, adaptation path, and probability mixture.
- Derived the scalar update equation, multidimensional MAP objective, Hessian, and probability-mixture Brier identity from the source.
- Compared the equations and counts on manuscript pages 3 and 4 against those paths.

## Claim-to-source register

| Claim | Source | Evidence class |
| --- | --- | --- |
| Public interface is `predict(input, labeled=None)`; there is no exported `Model` class. | `model.py:1103–1109` and AST inventory | OBSERVED |
| Two selected components are native CAIMIRA rank 2 and Rasch scalar-offset adaptation. | `model.py:1–546`, `550–1095`, selected state configurations | OBSERVED |
| Final output is the equal-weight arithmetic probability mixture. | `model.py:1103–1109` | OBSERVED |
| Evidence is filtered by exact subject-dictionary equality and matching populated benchmark. | `model.py:85–115`, `1103–1107` | OBSERVED |
| Each component accepts up to 31 revealed binary integer labels. | `model.py:85–115`, `481–523` | OBSERVED |
| Item features use complete item text; subject features use canonical JSON with sorted keys. | `model.py:30–70`, `85–115` | OBSERVED |
| Casefolded word/punctuation tokens enter a signed, L2-normalized, 384-dimensional hash map. | `model.py:22–35` | OBSERVED |
| BLAKE2b uses an 8-byte digest and personalization `roadmap.lex.v1`; first four bytes choose the bucket, fifth byte chooses the sign. | `model.py:22–28` | OBSERVED |
| Frozen item/subject features and forward passes use float32; the native MAP optimizer receives float64 arrays. | `model.py:30–35`, `129–142`, `293–391`, `481–523`; state dtype | OBSERVED |
| Active structured runtime constructs the model on CPU. | `model.py:394–531` | OBSERVED |
| Rasch correction solves sum_j(sigmoid(ell_j+delta)-y_j)+delta=0 through 42 bisections over [-n,n]. | `model.py:72–83`, `111–115` | OBSERVED |
| Rasch support logits are obtained after clipping frozen probabilities to [1e-9,1-1e-9]. | `model.py:114` | OBSERVED |
| Native CAIMIRA adaptation minimizes summed logistic loss plus one half squared Euclidean distance to the metadata prior. | `model.py:129–142`, `481–523` | OBSERVED |
| Native MAP uses L-BFGS-B, analytic gradients, gtol=1e-8, ftol=1e-13, and initialization at the metadata prior. | `model.py:129–142` | OBSERVED |
| CAIMIRA item relevance is a two-dimensional softmax and item difficulty is a two-dimensional centered linear feature map. | `model.py:235–285`; CAIMIRA state | OBSERVED |
| Metadata prior is a centered subject feature map, multiplied by a row-presence indicator. | `model.py:287–291` | OBSERVED |
| Frozen state orientation matrices are identities. | Both state `orientation` entries | OBSERVED |
| CAIMIRA stores 2,307 learned coefficients and Rasch stores 769, for 3,076 combined. | Both state parameter arrays | COMPUTED |
| Both states use schema `proxy-structured-inductive/v1`. | Both state `schema` fields | OBSERVED |
| State files contain configuration, parameters, centers, and orientation; the archive supplies no training run ledger, optimizer trajectory, training-set identifier, or training timestamp. | Archive manifest and both JSON top-level keys | OBSERVED |
| The exported candidate consumes revealed evidence; it exports no acquisition policy. | Entire source and AST inventory | OBSERVED |
| Frozen item parameters remain fixed during inference; only subject ability or scalar correction is adapted. | Active source paths above | OBSERVED |
| Full candidate inference succeeds in the evaluation environment. | Runtime execution still required | UNVERIFIED |
| A specific training recipe or training data generated these exact coefficients. | External provenance still required | UNVERIFIED |

## Active method account

The candidate predicts binary outcomes through two frozen feature-based item-response models. It maps complete item text and canonical subject metadata to signed lexical hash vectors of width 384. Each hash vector is L2-normalized. The CAIMIRA component maps item features to two centered difficulty coordinates and two softmax relevance weights, while a centered metadata map supplies a two-dimensional subject prior. Its logit is an intercept plus the relevance-weighted difference between subject ability and item difficulty. Revealed outcomes update the subject ability through logistic MAP estimation with identity prior covariance. The Rasch component has a scalar item difficulty and scalar metadata prior. It adjusts the frozen logits using a shared scalar offset with a unit Gaussian penalty. The final prediction averages the two component probabilities with equal weights.

The evaluator supplies the revealed observations. The exported function retains observations matching the query subject and benchmark and accepts at most 31 binary labels. Empty support uses each component's frozen prediction. Item feature maps, relevance weights, metadata maps, and centering constants remain fixed throughout adaptation.

## Counts and settings

| Quantity | CAIMIRA | Rasch |
| --- | --- | --- |
| Rank | 2 | 1 |
| Item feature width | 384 | 384 |
| Subject feature width | 384 | 384 |
| Difficulty coefficients | 768 | 384 |
| Relevance coefficients including bias | 770 | 0 |
| Prior coefficients | 768 | 384 |
| Intercept | 1 | 1 |
| Total learned coefficients | 2,307 | 769 |
| Intercept value | 0.0092009287327528 | 0.02624727413058281 |
| Prior kind | metadata | metadata |
| Stored dtype | float32 | float32 |
| Stored ridge setting | 0.01 | 0.01 |

The stored configuration also includes CAIMIRA L1 value 0.0001, floor cap 0.25, floor shrinkage 0.1, item-floor shrinkage 1.0, and the `unit_gaussian_mean_penalty` nuisance prior. The selected families have zero floor; several configuration fields and generic model-family branches remain dormant. These fields alone establish stored settings, while their role in the historical optimization requires training records.

## Mathematical checks

For Rasch support size n, the derivative of the scalar stationarity function equals 1 plus the sum of Bernoulli variances, so the root is unique. Each sigmoid-minus-label term lies in [-1,1], which bounds the root in [-n,n]. After 42 halvings, the midpoint error is at most n/2^42 in exact arithmetic. The implementation returns that midpoint.

For native CAIMIRA, the Hessian equals I plus the sum over revealed items of p_j(1-p_j) r_j r_j^T. It is positive definite, so the stated MAP objective is strictly convex. The implementation's optimizer guard concerns finite objective values and failed solves with gradient residual above 10^-6; it places no a residual bound on every successful solve.

For probabilities p_R and p_C and binary outcome y, the mixture squared error equals the mean component squared error minus (p_R-p_C)^2/4. This identity supports comparison with the component mean at a shared evaluation point; an empirical improvement over either individual component requires component predictions or an ablation.

## Material limitations

- Static source inspection supports the method description. End-to-end execution, binary-interface acceptance by Codabench, runtime cost, memory use, and environment compatibility require runtime evidence.
- Torch is imported by the candidate and is absent from the inspected local Python runtime. Candidate execution was skipped; no runtime success claim is supported by this audit.
- The archive contains generic support for other model families and dormant entity-model paths. The report should describe the selected CAIMIRA/Rasch paths.
- Training provenance, data partitions, hyperparameter selection, training costs, and seed robustness require external run artifacts.
- A scalar subject offset or two-dimensional subject update adapts outcome propensity from sparse evidence. An acquisition-strategy contribution requires a separate artifact.
