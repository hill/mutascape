# Search-ranking audit

## Assessment

The evidence does not establish reliable method selection for a new protein from DMS singles alone.
The manuscript reports its numerical results correctly.
Positive ranking correlations show some agreement, especially on GB1 and TrpB4.
They do not establish an advantage over additive landscapes or a fixed ranking from other proteins.

This distinction matters for the paper's main claim.
Mutascape supplies repeatable synthetic experiments under stated assumptions.
Its biological usefulness requires evidence that those experiments support conclusions on measured proteins.
Current search results provide mixed evidence for that stronger claim.
Distribution and Walsh agreement cannot replace this test.

The newer shuffled-data test also needs a narrower interpretation.
It retains the single-effect distribution and can preserve search difficulty under site permutations.
Its failure does not establish that DMS inputs provide no useful information.

The audit leaves the generator, manuscript and frozen study outputs unchanged.
The audit code and results sit on `codex/search-ranking-audit-2026-09-24`.

## Original manuscript results

The audit independently recalculates rankings from all 11,760 DMS-led search runs.
It reproduces every manuscript Kendall value to floating-point precision.
All runs spend their full query budget.
Another check matches 2,700 current public-model queries against the original fitness arrays exactly.
The public API refactor did not change these predictions.

| New queries | Mutascape | Additive | Empirical range | Fixed ranking |
|---|---:|---:|---:|---:|
| 48 | 0.207 | 0.213 | 0.206 | 0.164 |
| 96 | 0.582 | 0.551 | 0.592 | 0.523 |

These values average Kendall correlations across GB1, TrpB4, T7 and TEV.
The fixed ranking uses only the five development proteins.
It excludes every confirmation protein.

Across both budgets, Mutascape exceeds the fixed ranking by 0.051.
The paired protein-bootstrap 95% interval is [-0.049, 0.120].
Its difference from the additive control is 0.013, with interval [-0.017, 0.041].
Its difference from the empirical-range control is -0.005, with interval [-0.054, 0.045].
None establishes an advantage across proteins.
These intervals use four independent protein units, with both budgets retained within each unit.
They have limited precision.

### Does choosing a method help?

The audit selects the method with lowest synthetic mean regret, then measures that method's regret on the real landscape.
An exact synthetic tie receives equal weight across the tied methods.
The fixed ranking uses the same rule.
This analysis evaluates the practical consequence of using each ranking.

| New queries | Mutascape selection | Fixed selection | Difference |
|---|---:|---:|---:|
| 48 | 0.29496 | 0.29496 | 0.00000 |
| 96 | 0.27907 | 0.27997 | -0.00090 |

Lower measured regret is better.
At 48 queries, the selected methods have identical measured mean regret on each of the four proteins.
Across both budgets, the difference is -0.00045, with a paired protein-bootstrap interval of [-0.00218, 0.00130].
These results do not establish useful target-specific method selection.
The analysis compares saved search outcomes, without a fresh experimental deployment.
Its hindsight-best method is only a descriptive reference.

### Uneven evidence across proteins

GB1 and TrpB4 show substantial ranking agreement.
T7 has weak measured repeatability.
Its median split-half correlation is 0.130 at 48 queries and 0.137 at 96.
These calculations use every five-versus-five partition of the ten trials.
They measure repeatability with half the trials, rather than a ceiling for the full experiment.

All eight ML methods have identical measured mean regret on TEV at both budgets.
Their within-ML Kendall correlation is therefore undefined.
TEV cannot distinguish these ML methods under the tested protocol.
Removing the four non-ML methods does not generally remove the positive correlations on the other proteins.
The full-ranking result is therefore not solely an artefact of including random search.

The original eligible-pair counts also reproduce exactly.
At 48 queries, Mutascape matches 39/45 pairs on GB1, 44/50 on TrpB4 and 5/5 on TEV.
T7 has no eligible pairs.
At 96 queries, the counts are 39/39, 42/42, 6/14 and 5/5, respectively.
High percentages describe selected comparisons, rather than every ordering among the twelve methods.

The eligibility rule uses separate 95% bootstrap intervals for 66 pairs.
It does not control the chance of any false positive across all comparisons.
An extra sensitivity check uses exact paired sign permutations.
It assumes exchangeable method labels within each paired search trial.
At an unadjusted 5% threshold, TEV has no eligible pairs at either budget.
T7 has only one at 96 queries.
The bootstrap-based percentages are therefore sensitive to the eligibility rule.
The audit also applies Holm correction across all 66 comparisons.
No comparison passes that stricter rule.
With ten trials, the smallest two-sided permutation p-value is 2/1024.
That exceeds the first Holm threshold, 0.05/66.
This check exposes limited testing resolution, rather than proving that all methods perform equally.

## Newer panel

The newer experiment includes 20 nominally held-out assays.
Five already have their measured optimum among WT or supplied singles.
Every method then starts at the optimum, so measured regret cannot rank methods.
The remaining budgets cover 15, 11 and 10 assays, respectively.
They represent nine, six and five genotype-space groups.
The eligible panels differ, so their means do not isolate a causal effect of query budget.
Several assays share a group and must not count as independent proteins.

The following table reproduces the newer report's average of separate sample correlations.
It also corrects the fixed ranking to exclude the whole target genotype-space group.
It gives each remaining group equal training weight.

| Queries | Groups | Mutascape | Shuffled singles | Fixed ranking, group excluded |
|---|---:|---:|---:|---:|
| 48 | 9 | 0.319 | 0.311 | 0.295 |
| 96 | 6 | 0.292 | 0.238 | 0.302 |
| 192 | 5 | 0.258 | 0.257 | 0.381 |

Mutascape's differences from shuffled singles have intervals that cross zero at every budget.
The experiment therefore fails its recorded criterion for added value over that control.
Its group-bootstrap intervals describe this small panel and use the recorded group assignments.
They do not establish independence between every related protein family.

### Direct replay with the manuscript calculation

The audit reran 30,240 searches on every evaluable target at all supported budgets.
This covers measured landscapes and three samples each from Mutascape and the shuffled-single control.
Every saved per-sample correlation reproduces exactly.
The measured method means agree within 1.2e-16.

The table below averages each method's regret across samples before calculating its ranking.
This matches the manuscript calculation.
Means give each genotype-space group equal weight.

| Queries | Groups | Mutascape | Shuffled singles | Fixed ranking, group excluded | Fixed development ranking |
|---|---:|---:|---:|---:|---:|
| 48 | 9 | 0.380 | 0.337 | 0.295 | 0.301 |
| 96 | 6 | 0.310 | 0.251 | 0.302 | 0.332 |
| 192 | 5 | 0.258 | 0.275 | 0.381 | Not available |

The development ranking uses only the original five development proteins.
Their saved DMS-led experiments do not include 192 queries.

There is a positive result at 48 queries against the group-excluded fixed ranking.
The mean advantage is 0.085, with a group-bootstrap interval of [0.0003, 0.174].
This interval narrowly excludes zero under that calculation.
The comparison is exploratory, and the interval treats the fitted rankings as fixed.
Against the separate development ranking, the advantage is 0.078 with interval [-0.028, 0.197].
Neither comparison establishes a general advantage across budgets.

Against shuffled singles, the differences are 0.043, 0.059 and -0.017.
All three group-bootstrap intervals cross zero.
An extra bootstrap also resamples search trials and landscape samples within groups.
Its intervals are [-0.080, 0.153], [-0.115, 0.209] and [-0.128, 0.096], respectively.
This analysis reaches the same conclusion about the shuffled control.

At 192 queries, Mutascape trails the group-excluded fixed ranking by 0.123.
The group-bootstrap interval is [-0.248, -0.024].
The correlation among the eight ML methods alone is only 0.027.
The corresponding within-ML correlations are 0.272 at 48 queries and 0.209 at 96.
The larger panel therefore supplies little evidence for fine ranking distinctions between ML methods at the largest budget.

The method-selection test gives the following measured regrets.

| Queries | Mutascape selection | Fixed selection, group excluded | Fixed development selection |
|---|---:|---:|---:|
| 48 | 0.11732 | 0.10796 | 0.12034 |
| 96 | 0.14545 | 0.13646 | 0.14798 |
| 192 | 0.15547 | 0.14213 | Not available |

Lower regret is better.
Mutascape selects worse methods on average than the group-excluded fixed control at all three budgets.
It selects slightly better methods than the separate development control at 48 and 96 queries.
Every paired interval for these selection differences crosses zero.
The replay does not establish an improvement in real search performance from using Mutascape to choose a method.

### Problems with the newer analysis

The original fixed control excludes one assay but can retain another assay of the same protein.
It also weights assays equally during training, so five RBD assays can dominate the control.
Excluding the whole group reduces the fixed control's correlation from 0.370 to 0.295 at 48 queries.
The corresponding changes are 0.333 to 0.302 at 96 and 0.422 to 0.381 at 192.
The uncorrected comparison overstates the fixed control's advantage.

The newer report averages one correlation per synthetic sample.
The manuscript instead averages method regret across samples before calculating one correlation.
These operations can give different answers.
The direct replay records trial-level results and compares them using the manuscript rule.
The replay does not change any method, generator parameter or query budget.

The newer report drops an entire control column when any assay has an undefined correlation.
This removes the additive control from its summary.
An undefined ranking should remain visible as a limitation, with the number of evaluable targets.
A mean over fewer targets must not masquerade as a matched comparison.

The code's Gaussian-single control uses the target's single-mutant mean and standard deviation.
It uses no contact map or other structural data.
Its current structural label therefore describes the implementation incorrectly.
It is also not a control without target fitness information.

Several newer panel datasets appeared in earlier project evaluations.
Examples include CR6261, CR9114, Hsp90, COV107-23, H3 haemagglutinin and SARS-CoV-2 RBD.
The records support fixed settings before this scoring run.
They do not support a wholly new, untouched protein panel.

### What shuffled singles test

On a complete binary library, each site contributes one single-mutant effect.
Permuting site labels preserves the set of effects and the genotype graph.
Relabelling the pair coefficients in the same way preserves every corresponding fitness value.
The random pair model assigns the same probability law before and after that relabelling.
Methods that treat sites equivalently therefore have the same distribution of search outcomes.
Finite seeds and implementation tie-breaking can still differ.

The audit checks this identity numerically on all 256 genotypes of an eight-site model.
Missing genotype masks, multiple alleles and site-specific information can break the symmetry.
The argument therefore does not predict equality for every measured library.

The null tests whether assigning effects to particular sites adds information for this comparison.
It retains the target's distribution of single-mutant effects.
Failure to beat it cannot establish that this distribution has no value.
The fixed-ranking and additive comparisons remain separate tests of practical added value.

## Combination-calibrated Mutascape

The current manuscript's matched calibration experiment tests distributions and local spectra, rather than search rankings.
Its better C2ST values do not resolve the ranking question.
The earlier five-protein calibration study uses different domains and excludes known singles from search.
It reports positive ranking correlations but does not establish superiority for the current DMS-led workflow.

The newer calibration-size experiment currently contains only ParD3 results.
ParD3 starts with its measured optimum, so its search correlations are undefined.
That experiment cannot yet support a search-transfer claim.

## Consequence for the paper

Search-ranking transfer is central if the paper claims that synthetic method comparisons carry over to real directed evolution.
The present evidence does not validate that claim in general.
It does support some agreement under the tested protocols, with clear failures and weak control differences.

The software can still support conditional synthetic experiments, sensitivity studies and reproducible method development.
That contribution has a narrower scope than a validated substitute for measured benchmark landscapes.
The manuscript should make this distinction visible in its main claim, rather than only in limitations.

The next decisive experiment should test method-selection benefit on proteins excluded from all earlier development.
It should use a fixed ranking trained on separate proteins and an additive control.
A zero-pair sigmoid control would isolate the value of random pair interactions from the nonlinear fitness map.
The primary outcome should measure real search performance of the selected method.
C2ST, Walsh spectra and full-ranking correlations should provide supporting evidence.
More search repeats would reduce uncertainty within each protein, but more independent proteins remain necessary.

A Kendall p-value against zero does not answer whether Mutascape adds value over a control.
The [SciPy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html) defines that null and explains tie handling.
[Demšar's paper](https://www.jmlr.org/papers/v7/demsar06a.html) discusses comparisons of algorithms across datasets.
Here, paired protein or genotype-group comparisons address the relevant generalisation question.

## Validation

All 574 Python tests pass.
The repository lint check passes.
The web checks pass, including all 34 browser-unit tests and the production build.
The audit also checks vectorised Kendall calculations against SciPy on 100 tied and constant-vector examples.
Tied method selection and the binary site-permutation identity pass their focused checks.
The report passes the prose linter without hard violations.

## Reproduction

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m scripts.audit_search_rankings saved
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m scripts.audit_search_rankings replay --jobs 4
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m scripts.audit_search_rankings replay-report
```

The script writes audit tables under `results/search_ranking_audit_20260924/`.
It preserves all source results.
Its branch starts from code revision `1ea36d2` and reviews manuscript revision `178b9cc`.
