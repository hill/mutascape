# Review assessment and project changes

The review identifies real evaluation gaps.
The new controls weaken the development-set transfer claim.
They do not support every proposed explanation.

## Assessment

| Review claim | Assessment | Action |
|---|---|---|
| Pair agreement lacks a useful null | Valid | Added a fixed ranking that excludes every assay of the test protein |
| Random-search comparisons may explain most agreement | Testable, but the proposed count overstates their share | Repeated scoring without random-search pairs |
| Another protein's synthetic ranking is a useful control | Valid | Scored every other donor protein with equal donor and target weights |
| The main table hides baseline results | Valid | Added every generator at all four budgets |
| Nine wins against chance expectation eight establishes a tie | Invalid inference | Reported the full grid without treating correlated cells as independent trials |
| Potts is an exact readout ablation | Invalid | Ran a matched control with the same pair coefficients, coupling, and seeds |
| The abstract gives the most favourable spectral number | Valid | Removed that number and stated the mixed shape evidence |
| Filling 118 missing patterns causes the large CreiLOV advantage | Unsupported | Tested alternative fills on the same binary domain |
| The pilot model deserves a separate result | Valid | Added a reusable experimental generator and a matched same-assay singles control |
| The development collections provide independent confirmation | They do not | Recorded a prospective protocol and tested an unused family |
| Twenty unused protein families are readily available | Unsupported under the declared rules | Screened 160 local sequence tables and found one qualifying family |
| GraphFLA correlation one needs a saturation flag | Valid | Marked the table and plot and reported ranking resolution |
| Metric names and independent coverage need correction | Valid | Used Kendall correlation consistently and described 31 assay entries across 17 groups |

## Fixed rankings change the interpretation

The control averages measured regret within assays, then proteins, then training proteins.
It excludes every assay of the target protein across all collections.
Fitting on all 31 entries would leak the target's measured search results.
Both the control and each generator use the same bootstrap-eligible comparisons.

| Collection at 48 queries | Mutascape | Fixed ranking | Other-protein synthetic ranking |
|---|---:|---:|---:|
| SSMuLA | 89.5% | 91.9% | 73.9% |
| CombinGym | 69.1% | 84.0% | 71.3% |
| GraphFLA | 73.5% | 73.8% | 67.6% |

Mutascape exceeds the fixed ranking in four of twelve collection-budget comparisons.
This rejects a general claim that high pair agreement shows target-specific benchmark value.

Removing random-search pairs gives Mutascape scores of 88.0%, 64.7%, and 71.9%.
The fixed-ranking scores are 91.0%, 84.7%, and 71.1%.
Only 76 of 489 eligible SSMuLA pairs involve random search at 48 queries.
These pairs comprise 15.5 per cent, rather than one third.

These descriptive comparisons do not provide uncertainty intervals across new proteins.
The same proteins and method pairs recur across budgets.

## Matched readout experiment

Potts differs from Mutascape in pair construction, scaling, and coupling.
The new control fixes the unit pair coefficients, coupling 0.1, seeds, and measured single-mutant fitness.
It uses raw single effects and their scale without the fitted readout.
Removing the readout while keeping latent singles would change the supplied single-mutant fitness values.

The experiment covers all 31 development entries at all four budgets.
At 48 queries, pair agreement improves with the readout on all three collections.
Across twelve comparisons, the readout improves pair agreement in nine.
It improves Kendall correlation in four, with one tie.
This supports a conditional effect rather than a universal advantage.
The manuscript reports every result in a separate table.
The default remains unchanged because these results do not justify a universal replacement.

## A real generator change

The project now provides `mutascape.experimental.PilotCalibratedLandscape`.
It fits mutation-depth effects through a sigmoid from same-assay singles and a measured combination pilot.
It preserves wild type and singles exactly, rejects unsupported substitutions, and supports deterministic draws and parameter restoration.
The original pilot script now uses this implementation for its sigmoid burden model.

The implementation reproduces held-out values from all nine archived CreiLOV prototype draws.
A matched singles-only control removes the original full-DMS versus combinatorial-assay input confound.

| Median measure | Same-assay singles-only | Pilot generator |
|---|---:|---:|
| Raw Wasserstein distance | 0.439 | 0.062 |
| Logistic classifier AUC | 0.999 | 0.595 |
| Boosted classifier AUC | 0.999 | 0.619 |

The reduction in raw distance is about 86 per cent against matched inputs.
The review's 94 per cent figure also includes changing the input assay.
The pilot needs 48 measured combinations beyond singles.
Its 48-query search test therefore costs 96 combination measurements including calibration.
The 87.2 per cent pair score describes repeated comparisons on CreiLOV, not independent proteins.
The model remains experimental and separate from the default.

## Spectral sensitivity

The missing-cell check holds all 32,768 binary patterns fixed.
It fills 118 missing patterns using observed means, medians, minima, or maxima, separately for measured and synthetic landscapes.
Median Mutascape spectrum distance ranges from 0.052 to 0.061.
The control without pair effects ranges from 0.212 to 0.254.
This does not support attributing the advantage to filling alone.

The complete-subcube check changes the analysed domain.
DHFR and eqFP611 also change the protein and library.
Those comparisons expose limited generality but do not isolate the cause of CreiLOV's favourable result.
The abstract now states the mixed evidence without privileging its lowest distance.

## Prospective confirmation

The recorded protocol fixes model hashes, selection rules, budgets, seeds, classifiers, and reporting before scoring.
The local catalogue contains 160 sequence tables, not 160 independent protein families.
Many contain nucleotide sequences, repeated proteins, small libraries, or incomplete coverage.
Only ParB-Noc qualifies under the complete-library and unused-family rules.
The selected NBS assay contains all 160,000 combinations at four sites.
Its reference and source come from [Jalal and colleagues](https://doi.org/10.1016/j.celrep.2020.107928).

| Queries | Eligible pairs | Mutascape | Fixed ranking |
|---|---:|---:|---:|
| 48 | 36 | 100.0% | 66.7% |
| 96 | 46 | 95.7% | 67.4% |
| 192 | 41 | 92.7% | 82.9% |
| 384 | 28 | 96.4% | 85.7% |

Other generators also perform well, including perfect agreement at 48 queries.
This does not establish a unique Mutascape advantage.
Median spectral distance remains 1.033 and raw Wasserstein distance remains 1.388 fitness units.
Median logistic and boosted-classifier AUCs are 0.900 and 0.963.
The new family supports search comparisons while repeating the shape failure.
It is one prospective test under a local protocol, not an externally registered or blinded study.

## Reproduction and evidence

All new numerical outputs live in `results/review_response_20260923/`.
The manifest records source and generator hashes.
The selection table records every exclusion.

```sh
uv run python -m scripts.check_confirmation_selection
uv run python -m scripts.assess_review_nulls
uv run python -m scripts.run_readout_ablation --budgets 48 96 192 384 --jobs 4
uv run python -m scripts.run_review_confirmation --jobs 2
uv run python -m scripts.assess_confirmation_shape
uv run python -m scripts.check_walsh_filling
uv run python -m scripts.validate_pilot_generator
uv run python -m scripts.summarise_review_experiments
uv run python -m scripts.rebuild_preprint
```

The experiments use local measured datasets and saved development trials.
Checkpointed search commands skip completed targets.
The confirmation command rejects changed source or generator hashes.
The preprint rebuild replays frozen summaries and checks their numerical agreement.

## Verification

All 554 Python tests pass.
Ruff and Biome checks pass.
The web checks pass, including 34 tests and the production build.
The preprint rebuild reproduces every frozen development summary.
The pilot replay reproduces all saved held-out values and all 348 of 399 matching search comparisons.
The output checks require all four budgets before generating final tables.

The confirmation classifier check adds wild-type fitness to centred tensors before computing assay-scale features.
A direct comparison with lazy-oracle values checks this conversion.
The initial calculation omitted that offset and overstated classifier separation.
The corrected calculation leaves search, Wasserstein, and spectral results unchanged.
