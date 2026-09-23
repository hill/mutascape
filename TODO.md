# Mutascape priorities

Updated 2026-09-09.

## Purpose

Build a useful synthetic landscape generator for comparing MLDE and ALDE algorithms.
Measured singles anchor the assay. Seeded interactions create repeatable combinatorial difficulty.
Optional single completion supports queries across the protein.
Synthetic values do not predict the true fitness of unmeasured combinations.

Keep the tool small. Its core needs are a queryable oracle, a campaign interface, repeatable results, and evidence about benchmark behaviour.

## Completed

- [x] Preserve measured wild type, singles, and supplied combinations exactly.
- [x] Provide fixed seeded oracles and lazy genotype queries with `+` mutation labels.
- [x] Provide mutation domains, batch queries, campaign budgets, and saved oracle manifests.
- [x] Compare 12 search methods at 48, 96, 192, and 384 queries.
- [x] Assess method comparisons on SSMuLA, CombinGym, and GraphFLA.
- [x] Hold out target proteins when fitting the empirical prior.
- [x] Retain additive, Potts, and Rough Mount Fuji controls.
- [x] Measure CreiLOV histogram distances and projected Walsh spectra across 30 draws.
- [x] Add optional calibrated ESM-2 completion across all protein positions.
- [x] Validate ESM completion with hidden substitutions and hidden positions against simple baselines.
- [x] Label measured, imputed, and synthetic queries in the API and studio.
- [x] Freeze imputed values in manifests for restoration without model weights.
- [x] Add paired C2ST with logistic and boosted classifiers, AUC, and balanced accuracy.
- [x] Repair C2ST permutations and save reproducible CreiLOV replicate controls.
- [x] Keep generated documentation figures and API signatures checked against the code.
- [x] Trace the CreiLOV classifier failure to assay mismatch and a missing negative combinatorial trend.
- [x] Compare same-assay anchors, transferred burden terms, and small pilot-calibrated models.
- [x] Check the import refactor against tests and regenerated C2ST results.

ESM implementation and validation are in [single completion](docs/esm_completion.md).
The classifier results and provenance repair are in [the C2ST audit](docs/c2st_audit.md).

## Next work, in priority order

1. [ ] Validate the burden model on another protein before adding a public calibration API.
   Keep the singles-only default and its raw C2ST limitation visible.
   Treat 48-combination pilot calibration as a separate input contract.
   Require both adversarial and search checks before changing the default.
   Freeze features and model settings before each new assessment.
2. [ ] Add one command that rebuilds the current publication reference results and figures.
   Record source hashes, prior settings, seeds, and software versions.
   Replace the obsolete reference command that rebuilds older structural experiments.
3. [ ] Validate a small whole-protein MLDE/ALDE example through the public API.
   Compare random search with one learning method under equal budgets and several oracle draws.
   Include imputed positions, query provenance, runtime, and a saved result that another user can replay.
4. [ ] Test ESM completion on a second protein with hidden positions and substitutions.
   Retain mean and position-mean controls. Report failure cases before recommending completion as a default.
5. [ ] Reserve a new measured protein collection for confirmation of benchmark usefulness.
   Freeze generator settings first. Report method agreement alongside adversarial discrimination and known failures.

## Scope limits

CreiLOV raw C2ST gives near-perfect discrimination, despite close standardised histograms and Walsh spectra.
This is an open limitation, not a completed realism claim.
Replicate disagreement is an assay-specific reference and does not define a universal acceptance threshold.
Pilot calibration experiments do not establish transfer from single-mutant data alone.
See [the model diagnosis](docs/creilov_model_diagnosis.md) for the tested changes and refactor checks.

Oracle variation across draws measures benchmark sensitivity. It does not give calibrated biological uncertainty.
The existing campaign noise option is sufficient for simple noisy-search experiments.

The active plan excludes folding/activity models, structural higher-order terms, mechanistic assay models, and a landscape-to-strategy recommender.
It also excludes biological prediction metrics as primary success criteria and new integration frameworks without a user need.
Those additions would expand the project without establishing the current tool's usefulness.
