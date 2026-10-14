# Benchmark efficacy results

The corrected experiment completed 17,280 campaigns on four connected protein libraries.
Every campaign spent its budget, with no fallback queries or recorded warnings.
Method rankings agree more closely on these restricted libraries than in the earlier sparse-pool experiment.
This comparison changes the search domain, so it does not show a general improvement in the generator.

At 96 evaluations, generator averaging lowers mean selected-method regret from 0.0448 to 0.0186.
Its confidence interval includes no improvement.
Additive landscapes give better mean rank agreement than the default model.
Keep the public generator unchanged and present these results as exploratory evidence about benchmark design.

## Methods from the supplied paper

Maheshwari, Ivanov and El Haddad evaluate synthetic benchmarks through absolute performance discrepancy and relative method rankings.
Their [2024 paper](https://arxiv.org/abs/2409.11968) also averages performance over several generators.
We adopted three parts of that evaluation.

| Method | Adaptation |
|---|---|
| Mean squared performance difference, MSPD | Average the squared difference between measured and synthetic mean regret across the twelve methods |
| Spearman correlation | Compare measured and synthetic method rankings, retaining ties |
| Generator averaging | Average method regret over sigmoid, linear and pooling models with equal weight |

The ensemble averages method performance, rather than genotype fitness.
Its conditions and weights were recorded before the searches ran.
Kendall correlation remains a secondary ranking summary.
Measured regret after method selection remains the primary outcome.
Changing a correlation metric therefore cannot change the primary success criterion.

The paper's self-generation bias factor concerns models that both generate benchmarks and solve the evaluated tasks.
Mutascape does not perform the search, so that statistic has no direct counterpart.
We instead save centred method performance across generator families as a descriptive preference check.
The paper does not supply our bootstrap procedure.

## Corrections and experimental design

The [preceding code audit](usefulness_code_audit_2026-09-30.md) identified unequal forest targets for tied fitness values.
The corrected forest uses average ranks.
The current protocol also excludes known WT and singles from each method's initial random design.
Model-guided methods receive 32 new combinations before choosing further batches.
Random search samples unknown candidates only.
The implementation records uniform random fallback if a method stops early.

Each of GB1, TrpB4, CreiLOV and eqFP611 supplies one connected library of 4,096 measured genotypes.
A seeded breadth-first traversal starts from WT and all supplied singles.
It uses genotype coverage without consulting fitness.
All four candidate graphs have one component and no isolated genotype.
This preserves local paths but favours genotypes near WT.
These libraries do not represent the full measured landscapes.

Every method receives the same candidates, known observations and budget within each protein.
Measured fitness never enters generator fitting beyond WT and singles.
The study uses five synthetic landscape samples with six search repeats per sample.
Each measured condition receives thirty repeats.
The twelve methods run at budgets of 48 and 96 new evaluations.
The default, zero-pair sigmoid, additive, linear and pooling conditions retain their earlier hyperparameters.

The primary comparison uses 96 evaluations.
The fixed control chooses a method from measured performance on the other three proteins.
Tied choices and proteins receive equal weight.
Five thousand bootstrap samples repeat method selection and retain pairing between generator conditions.
Aggregate contrasts also resample proteins.
These intervals describe computational sampling variation, without experimental measurement uncertainty.

The [protocol](benchmark_efficacy_protocol_2026-09-30.md) and source hashes were recorded before search outcomes were available.
The new study does not retrospectively correct the archived studies.
The manuscript now uses this separate experiment for its search figures and results.

## Main results

Values below average the four proteins equally at 96 evaluations.
Lower MSPD and regret are better.
Higher correlations indicate closer method rankings.

| Benchmark | MSPD | Spearman | Kendall | Selected-method regret | Fraction of improvement missed |
|---|---:|---:|---:|---:|---:|
| Mutascape | 0.00894 | 0.730 | 0.652 | 0.04481 | 0.05616 |
| Zero-pair sigmoid | 0.00902 | 0.763 | 0.691 | 0.04481 | 0.05616 |
| Additive | 0.00672 | **0.824** | **0.730** | 0.02520 | 0.03271 |
| Linear | **0.00551** | 0.764 | 0.627 | 0.02364 | 0.03124 |
| Pooling | 0.00617 | 0.762 | 0.640 | **0.01861** | **0.02497** |
| Ensemble | 0.00626 | 0.760 | 0.638 | **0.01861** | **0.02497** |
| Fixed choice | 0.00685 | 0.627 | 0.437 | 0.02856 | 0.03630 |

The ensemble difference from Mutascape is -0.02620, with a 95% interval of [-0.09022, 0.01292].
Its difference from the fixed control is -0.00995, with an interval of [-0.04167, 0.02972].
Both intervals include zero.
Pooling alone selects the same measured outcomes as the ensemble.
Averaging does not show an advantage over the best individual family here.

The largest selection change occurs on eqFP611.
Mutascape selects greedy local search, with measured regret 0.1770.
The ensemble selects pairwise ridge search, with measured regret 0.0720.
At 48 evaluations, ensemble regret is 0.09463, compared with 0.09165 for Mutascape.
The apparent advantage therefore depends on evaluation budget.

| Protein | Mutascape Spearman correlation at 96 evaluations |
|---|---:|
| GB1 | 0.894 |
| TrpB4 | 0.957 |
| CreiLOV | 0.706 |
| eqFP611 | 0.364 |

A sensitivity analysis retains only the eight model-guided methods.
It was added after inspecting the main results.
Mean Spearman correlation then becomes 0.592 for Mutascape, 0.608 for the ensemble and 0.724 for additivity.
Some agreement therefore reflects differences between broad search families.

## What remains unresolved

All generated conditions leave some opportunity to improve beyond known observations.
This removes the completely tied rankings caused by zero available improvement in the earlier restricted-pool experiment.
It does not establish a realistic search difficulty.

On eqFP611, 43.7% of generated unknown combinations improve on the best supplied observation, compared with 6.0% of measured combinations.
The mean maximum improvement is only 0.0096 for Mutascape, compared with 1.2524 for measured fitness.
On CreiLOV, the corresponding improving fractions are 37.1% and 2.4%.
The generator can reproduce some method rankings while misrepresenting the useful fitness tail.

This study uses one library per protein and only four proteins.
The results remain exploratory.
Connected library selection, corrected initialisation and protein composition differ from the earlier studies.
The stronger correlations cannot be attributed to one correction in isolation.
The study does not establish reliable method selection on a new protein or across a complete landscape.

The current results support controlled simulated campaigns under explicit interaction assumptions.
They do not show that random pair effects improve method ranking over simpler controls.
New evaluation should retain the present protocol and use several independently chosen libraries per protein.
That would test whether the apparent utility persists beyond one local region.

## Files and reproduction

Results reside in `results/benchmark_efficacy_20260930/`.

| File | Content |
|---|---|
| `manifest.json` | Protocol, libraries, seeds, settings and input hashes |
| `trials.csv` | All 17,280 campaign outcomes |
| `per_protein.csv` | Metrics and selected methods |
| `intervals.csv`, `contrasts.csv` | Per-protein intervals and aggregate paired contrasts |
| `headroom.csv` | Available improvement and improving fractions |
| `centred_method_performance.csv` | Descriptive method preference by generator |
| `ml_only_sensitivity.csv` | Secondary analysis of model-guided methods |
| `audit.json` | Budget checks, fallback counts and output hashes |

The runner verifies frozen inputs before resuming saved experiments.
The analysis command recalculates metrics from the saved trials.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m scripts.benchmark_efficacy prepare
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m scripts.benchmark_efficacy run --jobs 4
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m scripts.analyse_benchmark_efficacy
.venv/bin/python scripts/rebuild_reference_outputs.py
```

The manuscript builder now reads these corrected results instead of the historical search table.
The reference rebuild entry point now targets the current manuscript rather than the old structural-prior figures.
The public generator and its defaults remain unchanged.
All manuscript edits remain uncommitted for review.

## Validation

The full Python suite passes all 607 tests.
Ruff, browser linting, type checks, all 34 browser tests and the browser build pass.
Generated documentation assets match their source.
The current manuscript rebuild succeeds, and its figures and tables pass visual inspection.
The new prose passes the project style checker without hard violations.
