# Mutascape priorities

Updated 2026-09-23.

## Purpose

Build useful synthetic landscapes for comparing MLDE and ALDE algorithms.
The manuscript model uses same-assay wild type, singles and a random measured combination pilot.
Seeded interactions and a percentile map create repeatable combinatorial fitness values.
Synthetic values do not predict the true fitness of unmeasured combinations.

## Completed

- [x] Provide fixed seeded oracles, mutation domains, campaign budgets and saved manifests for the singles-only interface.
- [x] Compare 12 search methods across SSMuLA, CombinGym and GraphFLA at four query budgets.
- [x] Correct grouped protein summaries and exclude every target-protein assay from prior fitting.
- [x] Add fixed rankings from other proteins as a control.
- [x] Add optional ESM-2 single completion and validate hidden substitutions and positions.
- [x] Repair paired C2ST permutations and record CreiLOV replicate controls.
- [x] Trace the CreiLOV mismatch to assay differences and a missing combination-depth trend.
- [x] Implement pilot and quantile generators with same-assay input checks and exact preservation of wild type and singles.
- [x] Evaluate the quantile model on five proteins with 48 and 192 pilot combinations.
- [x] Audit historical genotype reuse and rerun affected CreiLOV results.
- [x] Report classifier, distribution, spectral and search results together.
- [x] Compare the final model with Additive, Potts, RMF and two NK settings on identical search domains.
- [x] Restore the manuscript dataset and generator tables, with the best displayed values in bold.
- [x] Retain four figures and add error bars to the search and CreiLOV results.
- [x] Rebuild the manuscript around one final model and retain the required abstract claim.

The [project status](docs/project_status.md) gives current results and limitations.
The [quantile model guide](docs/quantile_generator.md) describes inputs and the API.
The [result report](docs/quantile_calibration_2026-09-23.md) records development and confirmation experiments.

## Next work

1. [ ] Test previously unstudied protein families under a fixed protocol.
2. [ ] Repeat pilot selection to measure variation across calibration samples.
3. [ ] Compare pilot fitting and direct optimisation with equal total measurement budgets.
4. [ ] Validate ESM completion on a second protein with hidden positions and substitutions.
5. [ ] Connect the accepted model to the browser and campaign interfaces when their input workflows support measured pilots.

## Scope limits

Classifiers still distinguish generated and measured landscapes.
Matching fitness distributions or order-level variance does not establish correct genotypes, peaks or interacting substitutions.
The search results describe five proteins and the tested methods and budgets.
The model has one calibration set per protein, with three conditional generator draws.
Draw variation measures sensitivity to generator assumptions, rather than biological uncertainty.

The singles-only API remains available, but its benchmark results are separate from the manuscript model's results.
