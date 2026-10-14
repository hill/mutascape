# Full-dataset C2ST results

The full evaluation uses every experimental multi-mutant genotype in the supported search space except the 96 calibration genotypes.
Mutascape and Combination-calibrated Mutascape use identical evaluation genotypes and folds within each protein.
The generators retain their existing fitness arrays and hyperparameters.

## Results

Values below are sequence-aware AUCs, averaged across five validation folds and then summarised by the median across three landscape seeds.
An AUC nearer 0.5 indicates weaker discrimination by this classifier.

| Protein | Evaluation genotypes | Mutascape | Combination-calibrated Mutascape |
|---|---:|---:|---:|
| GB1 | 149,188 | 0.996 | 0.811 |
| TrpB4 | 158,956 | 0.991 | 0.766 |
| T7 | 2,699 | 0.993 | 0.564 |
| TEV | 158,959 | 0.950 | 0.784 |
| Mean | | 0.982 | 0.731 |

The corresponding means for the historical 1,000-genotype sample were 0.979 and 0.687.
Combination calibration gives lower AUC on all four evaluation proteins, but the full-data classifiers still detect differences.
This evaluation uses new seeded fold assignments as well as more data, so the change cannot be attributed solely to sample size.

The five datasets used for model and hyperparameter selection give the following results.

| Protein | Evaluation genotypes | Mutascape | Combination-calibrated Mutascape |
|---|---:|---:|---:|
| DHFR | 7,846 | 0.992 | 0.778 |
| eqFP611 | 8,082 | 0.992 | 0.842 |
| ParB-Noc | 159,827 | 0.956 | 0.717 |
| ParD2 | 7,728 | 0.810 | 0.806 |
| CreiLOV | 165,311 | 1.000 | 0.814 |

## Interpretation

The 1,000-genotype sample limited computation and gave classifiers equal training sizes across proteins.
The full-data evaluation is feasible and uses all eligible measurements for comparisons within each protein.
Training sizes now differ across proteins, which can affect classifier performance.
AUC near chance does not establish that distributions are equivalent.

Wasserstein distances, Walsh spectra and search-method results are unchanged.
C2ST measures distinguishability and does not test whether a generator selects better search methods.
There is still one calibration set per protein, so variation across landscape seeds excludes uncertainty from choosing that set.

## Reproduction

Run from the code repository.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python -m scripts.evaluate_full_c2st prepare
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python -m scripts.evaluate_full_c2st run --jobs 3
uv run python -m scripts.evaluate_full_c2st summarise
uv run python -m scripts.build_review_manuscript --tables-only
```

Results are in `results/manuscript_full_c2st_20261006`.
The protocol records classifier parameters and source hashes.
Each protein directory contains the genotype groups, validation folds and results for both feature sets.
The completed run contains 108 C2ST estimates and 540 classifier fits.
The original execution source and protocol are retained alongside a record of one later formatting change to a status message.

The updated manuscript summaries replace C2ST values only for the two generators reported in its distribution table.
Other comparator records retain their historical C2ST values.
The earlier sampled results remain available for comparison.
