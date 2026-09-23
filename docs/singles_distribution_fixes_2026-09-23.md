# Simple checks of the CreiLOV distribution failure

The upper spike reflects two assumptions in singles-only Mutascape.
Its sigmoid limits come from a narrow set of single-mutant values.
Most supplied CreiLOV single effects are positive, so combinations approach the upper limit.

These exploratory checks followed inspection of that failure.
They use CreiLOV and the four proteins from the earlier confirmation panel.
Those proteins are no longer independent confirmation for any setting chosen from this new comparison.
No target combination measurements fit the candidate models.
No candidate replaces the manuscript model.

## Fixed changes

The comparison retains the existing 1,000 evaluation genotypes, 256 complete three-site subcubes and three landscape seeds per protein.
All candidates use pair strength 0.3.
Grouped C2ST uses the same classifier, features and folds as the manuscript.
The sequence-aware results below first take the median across landscape samples.

The changes are

- Wider sigmoid limits, with the margin increased from 5% to 50% of the singles range.
- A linear fitness map, with no sigmoid limits.
- The linear map with a negative mean pair contribution.

The last candidate subtracts `0.3 * single_effect_SD * d * (d - 1) / 2` for a genotype with `d` substitutions.
Its random pair coefficients remain unchanged.
The penalty is zero for WT and singles, so every supplied measurement remains exact.
The coefficient 0.3 was fixed before these checks, without a parameter sweep.
Its negative sign is an added assumption and cannot be inferred from singles alone.

Nonlinear losses after accumulating mutations occur in GFP experiments.
This motivates testing mutation penalties, but does not establish their magnitude or transfer to other proteins.
See [Sarkisyan et al.](https://www.nature.com/articles/nature17995) and [Otwinowski et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC6094095/).

## Results

| CreiLOV model | Sequence-aware C2ST AUC | Wasserstein / measured SD | Local Walsh distance |
|---|---:|---:|---:|
| Mutascape | 1.000 | 1.508 | 0.028 |
| Wider sigmoid | 1.000 | 1.781 | 0.079 |
| Linear map | 1.000 | 2.269 | 0.138 |
| Linear map with negative pair mean | 0.800 | 0.191 | 0.140 |

The negative pair mean improves CreiLOV's distribution while worsening its local spectral agreement.
It also produces worse average results on the other four proteins.

| Mean over GB1, TrpB4, T7 and TEV | Sequence-aware C2ST AUC | Wasserstein / measured SD | Local Walsh distance |
|---|---:|---:|---:|
| Mutascape | 0.979 | 1.116 | 0.500 |
| Wider sigmoid | 0.984 | 2.829 | 0.682 |
| Linear map | 0.986 | 4.700 | 0.736 |
| Linear map with negative pair mean | 0.995 | 6.866 | 0.739 |

The negative pair mean worsens C2ST and distribution distance on each of those four proteins relative to Mutascape.
Removing the sigmoid already worsens their fitness ranges.
Adding a penalty then compounds errors for proteins whose single effects are largely deleterious.
A universal negative penalty is therefore a poor replacement.

The next useful test would separate assay-range assumptions from mutation-burden assumptions.
It should select any range or penalty rule on other proteins before evaluating the target.
Known assay limits could provide range information where they exist.
A fluorescence-specific penalty might help, but this experiment does not validate one.
Combination-calibrated Mutascape remains the tested route when target combination measurements are available.

These checks did not repeat the 12-method search comparison.
The results do not establish better search benchmarks or justify changing the manuscript's default model.

## Reproduction

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python -m scripts.check_singles_distribution_fixes --jobs 4
```

The command records its protocol before evaluation.
Raw metrics and protocol hashes are in `results/singles_distribution_fixes_20260923`.
The compact summary is `src/mutascape/data/validation/singles-distribution-fixes-v1.json`.
