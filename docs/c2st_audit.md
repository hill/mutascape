# CreiLOV adversarial assessment

Date: 2026-09-09

The current generator fails the CreiLOV assay-scale classifier test.
Held-out classifiers separate measured and synthetic values with AUC above 0.999.
This agrees with the raw Wasserstein error of 3.94 measured standard deviations.
The close standardised histogram and Walsh spectrum do not imply agreement on the assay scale.

## Repair of the previous floor claim

Commit `6dcb1cd` claimed a real-replicate C2ST median of 0.699.
Its code change added one argument to a diagnosis script.
It did not save the claimed replicate calculation or its outputs.
The stored `output_260820_c2st_floor/summary.json` instead reports eight assay proxies with median AUC 0.8200625.

Those proxies include different phenotypes and reused GB1 measurements.
They cannot estimate a replicate noise floor.
The old script now labels these comparisons as proxies and directs readers to the replicate audit.
Neither 0.699 nor 0.820 should act as a threshold for accepting synthetic landscapes.

Even genuine replicates do not define a universal lower bound.
Their discrimination depends on sample size, measurement processing, features, and classifier capacity.
A deterministic generator also differs from a noisy assay replicate and from an average of replicates.

## Protocol

The source is CombinGym's raw CreiLOV workbook, which contains three fluorescence replicates.
The script checks the supplied logarithms against log10 fluorescence.
It verifies the mean-log values against the prepared combinatorial dataset on 165,407 shared multi-mutants.
It subtracts each assay's own measured wild-type log fluorescence before comparison.
This removes multiplicative brightness offsets but retains differences in shape and mutation-depth dependence.

The generator uses measured single effects and a prior fitted without CreiLOV.
No combinatorial value enters generator fitting.
The audit uses the same strict generator as the existing shape analysis, without ESM completion.

1. Sample 2,000 matched multi-mutant genotypes with each of three fixed sampling seeds.
2. Score three fixed synthetic draws and a control without pair interactions.
3. Build the same features for each assay and synthetic draw.
4. Assign every genotype pair to one of five folds.
5. Fit each classifier on four folds and score the remaining fold.
6. Swap source labels within genotype pairs for 99 permutation tests on the first sampling seed.

Features include fitness, additive expectation, their residual, squared fitness, absolute residual, squared residual, and mutation depth.
Fitness features share a scale from measured DMS singles.
They do not use separate whole-distribution standardisation.
Logistic regression scales its features using training folds only.
The second classifier uses histogram gradient boosting with seven leaves, 100 iterations, and no early stopping.
Both classifier settings stay fixed across comparisons.

The audit reports mean fold AUC and mean fold balanced accuracy separately.
AUC scores probability ordering across classes. Balanced accuracy scores decisions at probability 0.5.
Neither score is a biological prediction metric.
The paired permutation p-value is `(1 + exceedances) / (1 + permutations)`.
With no permutations, the p-value is unavailable.

The old test shuffled labels across genotypes.
That permutation broke the paired sampling design.
The revised test retains one label of each class per genotype and refits the classifier on every permutation.

## Results

Values below are medians across three genotype samples.
The synthetic row includes all three fixed draws.

| Comparison | Logistic AUC | Boosted-tree AUC | Logistic balanced accuracy |
|---|---:|---:|---:|
| Replicate 1 versus 2 | 0.511 | 0.517 | 0.511 |
| Replicate 1 versus 3 | 0.661 | 0.667 | 0.619 |
| Replicate 2 versus 3 | 0.654 | 0.666 | 0.608 |
| Measured versus synthetic | Above 0.999 | Above 0.999 | Above 0.998 |
| Measured versus identical copy | 0.500 | 0.500 | 0.500 |

The no-pair control also gives AUC 1.000.
Replicate 3 retains a measurable difference after wild-type normalisation.
This does not establish its cause as a batch effect.
Replicates 1 and 2 show weak discrimination, but the logistic permutation test still gives p = 0.01 on the first sample.
The boosted classifier gives p = 0.10 for that pair.
Small effects can therefore be statistically detectable.

Synthetic comparisons give p = 0.01, the resolution limit with 99 permutations.
These are individual diagnostic p-values without correction for multiple comparisons.
The large effect sizes drive the conclusion here.
Sample ranges measure sensitivity to the selected genotypes, not confidence intervals across proteins.
The three genotype samples can overlap.

A smaller check with 200 genotypes gives logistic replicate AUCs of 0.509, 0.654, and 0.652.
It does not reproduce the historical 0.699 claim under this explicit protocol.

## Reproduction

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python scripts/audit_creilov_c2st.py \
  --output results/creilov_c2st --permutations 99

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python scripts/replay_c2st_audit.py \
  results/creilov_c2st
```

The first command reads the raw workbook and regenerates all inputs and results.
The second recomputes every fold's AUC and accuracy from frozen feature matrices.
It needs no source workbook or model weights.
Metadata records input hashes, model parameters, sampling seeds, software versions, and code hashes.
The documentation chart reads these result tables through `just docs-assets`.

## Next adversarial test

The [model diagnosis](creilov_model_diagnosis.md) traces the assay mismatch and tests small model changes.
It separates singles-only transfer from experiments that use measured combinatorial pilot batches.

Keep the assay-scale C2ST as a calibration diagnostic.
Add a separate test of local neighbourhoods after a declared fitness transformation.
Give its classifier neighbour fitness differences and uphill fractions from matched genotype patches.
Keep overlapping patches within the same fold, or use disjoint patches.
This tests local search structure that a histogram cannot describe.

A later trajectory classifier could distinguish complete search traces on measured and synthetic landscapes.
Hold out whole campaigns and proteins for that test.
Its features could include improvement per query and distances between successive variants.
This would test the intended benchmarking use more directly.

Freeze those tests before changing the generator.
Passing a weak discriminator does not establish equivalence.
Retain independent measured proteins for the next confirmation study.
The [C2ST paper by Lopez-Paz and Oquab](https://arxiv.org/abs/1610.06545) gives the classifier-based testing framework.
