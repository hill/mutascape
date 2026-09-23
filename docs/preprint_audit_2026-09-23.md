# Preprint audit

Date: 2026-09-23

## Outcome

The initial audit reproduced the archived protein-balanced search scores at all four budgets.
A follow-up corrected GraphFLA prior leakage and replaced the affected trials.
The CreiLOV distribution distances and classifier results also reproduce.
The original text overstated protein holdout and spectral realism.
The revised preprint reports these limits and explains each score directly.

The checkpoint commits are `f4499ca` in Mutascape and `b0c5306` in the manuscript repository.
Both use the subject `23-09-26 checkpoint`.
They preserve the work present before this audit.

## Search results

The audit recalculated rank correlations, paired-bootstrap decisions, and split-half reference values from saved trials.
All 12 collection-budget combinations reproduced their saved tables.
The recalculated protein-balanced scores matched the frozen JSON for every reported stochastic generator.

SSMuLA trials and six unaffected GraphFLA landscapes came from the archived research worktree.
New GraphFLA eqFP611 trials exclude all three CombinGym eqFP611 readouts from prior fitting.
The CombinGym trial tables were absent from the available project directories.
New runs recreated them at 48, 96, 192, and 384 queries with all 12 methods.
Each run used ten trials, three generator samples, coupling 0.1, and seed 20260823.
The resulting summaries matched the frozen values.

| Collection | Queries | Kendall correlation | Pair agreement | Eligible pairs |
|---|---:|---:|---:|---:|
| SSMuLA | 48 | 0.556 | 89.5% | 489 |
| CombinGym | 48 | 0.364 | 69.1% | 222 |
| GraphFLA | 48 | 0.439 | 73.5% | 191 |

These percentages average matching fractions across proteins.
They are neither pooled fractions across every assay nor probabilities of selecting the best method.
The revised table reports every budget and its eligible pair count.

At 384 queries, all measured methods find the CR6261 optimum in every trial.
Its rank correlation and measured split-half reference are undefined.
The summary excludes it from rank and repeatability calculations, and it contributes no eligible method pairs.

The GraphFLA median Kendall correlation of one does not mean that every landscape agrees.
Four landscapes give one, while the other three give 0.262, 0.418, and 0.563.
The revision states this directly instead of attributing the median to search saturation.

CSV replay uses round-trip float parsing.
The default parser changed one numerical tie in an older, non-headline posterior comparison.
Round-trip parsing reproduced the saved correlation exactly.

## Spectral checks

Thirty new CreiLOV draws reproduced the following values.

| Quantity | Recalculated value |
|---|---:|
| Median standardised Wasserstein distance | 0.046713 |
| Median raw Wasserstein distance | 0.947695 |
| Raw distance in measured standard deviations | 3.943173 |
| Median projected Walsh distance | 0.056836 |
| Walsh distance without pair terms | 0.254 |
| Walsh distance across draws, 5th to 95th percentiles | 0.020422 to 0.107602 |

The binary projection has 32,650 observed patterns out of 32,768.
The existing implementation fills 118 missing patterns with the mean of observed pattern means.
It averages amino-acid variants within a pattern before giving every binary pattern equal weight.
The revision states both operations.

A coverage-only search selected the largest complete subcube, with 8,192 patterns.
It fixes positions 47 and 113 as mutated.
Its median Walsh distance is 0.111926, versus 0.126163 without pair terms.
This comparison changes the domain as well as removing filling.
It does not isolate imputation bias.

Additional checks used complete DHFR and eqFP611 red libraries.
Neither needed filling or an amino-acid projection.
Prior fitting excluded the target protein, and the generator used only its wild type and singles.
These are existing development datasets, not independent confirmation proteins.

| Dataset | Measured combinations | Median Walsh distance | Without pair terms |
|---|---:|---:|---:|
| DHFR | 8,000 | 0.605315 | 0.641099 |
| eqFP611 red | 8,192 | 1.045720 | 1.134641 |

Both synthetic libraries retain excess additive variance.
The revision therefore limits the close-spectrum claim to the CreiLOV projection.
It retains method comparison as the main test of benchmark usefulness.

The new explanatory figure uses four exact values, with a mean coefficient of four and nonconstant coefficients of minus two, minus one, and one.
Its additive and pair variance fractions are five-sixths and one-sixth.
The figure script checks reconstruction and variance accounting against the package transform.
The manuscript cites Brookes et al. for the Fourier basis and Heckendorn and Whitley for the bit-mask interpretation.
The figure labels genotype bits separately from coefficient masks.
The number of selected bits gives the interaction order.

## Other checks and corrections

- Replayed every fold's AUC and balanced accuracy for all 48 saved C2ST comparisons.
- Checked paired grouping, training-only logistic scaling, and within-pair permutation code.
- Corrected the permutation description, which applies to the first genotype sample only.
- Recalculated ESM validation metrics from all saved held-out predictions.
- Checked the 2,121 measured singles, 118 measured positions, and 119-residue sequence through the public API.
- Checked the shared-single assay correlation of 0.131295 across 19 substitutions.
- Corrected the landscape table to report searchable candidates consistently.
- Corrected the Potts coupling to 0.25 and stated that RMF noise also changes supplied fitness values.
- Described the local AdaLead-style implementation and the different surrogate feature representations.
- Removed an unsupported claim about Wilcoxon tests from the reported uncertainty analysis.
- Removed claims of exact affine invariance and complete archived provenance.

Every collection now excludes the target protein from prior fitting.
The GraphFLA correction removes all three CombinGym eqFP611 readouts when testing eqFP611.
It reruns all 12 methods, ten trials, and three generator samples at each budget.
The other six GraphFLA targets had no training overlap, so their trial values remain unchanged.
Measured, additive, Potts, and RMF results reproduce for eqFP611, which checks seed and candidate consistency.
The corrected panel also uses current protein labels and a consistent source field.
Each corrected directory records prior training datasets and the hashes of both trial sources.

Subset selection previously changed dataset seed indices.
Both GraphFLA and within-protein subset runs now preserve the indices from the full panel.
Three regression tests cover protein exclusion and seed stability.

The corrected GraphFLA pair agreement is unchanged at 48, 96, and 384 queries.
At 192 queries, it falls from 82.1% to 80.4%.
Median rank correlations, eligible pair counts, and measured repeatability are unchanged at every budget.
The manuscript, figures, frozen summary, and documentation now use the corrected trials.

The API example ran successfully through `from_dms` and a fixed oracle draw.
No model parameters or public API behaviour changed during this audit.

## Artefacts

| File | Purpose |
|---|---|
| `results/preprint_audit_20260923/search_summary.csv` | All collection, budget, and baseline scores |
| `results/preprint_audit_20260923/eligible_pairs.csv` | Matching and eligible counts by landscape |
| `results/preprint_audit_20260923/draw_sensitivity.csv` | Scores for each synthetic draw |
| `results/preprint_audit_20260923/search_provenance.json` | Trial locations and file hashes |
| `results/preprint_audit_20260923/combingym_b*/` | Recreated CombinGym trials and summaries |
| `results/preprint_audit_20260923/creilov/` | Rebuilt distribution and projected-spectrum results |
| `results/preprint_audit_20260923/creilov_projection.json` | Pattern coverage and complete-subcube selection |
| `results/preprint_audit_20260923/complete_spectra.csv` | Full DHFR and eqFP611 variance fractions |
| `results/preprint_audit_20260923/imputation_replay.csv` | Recalculated ESM validation errors |
| `scripts/plot_walsh_explainer.py` | Exact, reproducible explanatory figure |

## Reproduction

From the Mutascape root, run the following command.

```sh
uv run python -m scripts.rebuild_preprint
```

This replays saved search statistics and classifier scores, reruns the spectral checks, rebuilds figures and the result table, and compiles the manuscript.
It needs the prepared assay files, archived SSMuLA trials, corrected GraphFLA trials, and recreated CombinGym trials listed in the provenance file.
It does not rerun the full search campaigns.
Use `--archive`, `--results`, and `--manuscript` to change those locations.
The command records audit code hashes and software versions.
Those hashes describe this audit, not the original archived runs.

To recreate a CombinGym search run, change the budget and output path as needed.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python scripts/run_search_transfer.py \
  --ssmula-root .mutascape/benchmark-data/ssmula \
  --combingym-root .mutascape/benchmark-data/combingym \
  --target-source combingym --generators primary \
  --budget 48 --seeds 10 --draws 3 --jobs 4 \
  --output results/preprint_audit_20260923/combingym_b48
```

To recreate the GraphFLA correction, run the following command.
It checks the unaffected controls before replacing eqFP611 trials.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python -m scripts.repair_graphfla_holdout --rerun
```

## Limits

The audit did not rerun SSMuLA or the six GraphFLA landscapes without prior overlap.
It reproduced their summaries from trial-level regrets and reran GraphFLA eqFP611.
The classifier replay reproduced effect sizes without repeating all permutation fits.
ESM replay used saved predictions and did not rerun the language model.
No new protein was reserved for independent confirmation.
The next scientific test should freeze the model and select such a protein before viewing its results.

## Final checks

The Python suite passed all 546 tests.
The web suite passed all 34 tests.
Lint, type checks, documentation asset checks, and the web build passed.
The manuscript compiled, and all 11 pages passed visual review.
The PDF contains no instances of the removed terminology.
Template author names and an unsupported funding statement were removed.
The actual author list and funding details still need confirmation before submission.
