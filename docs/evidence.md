# Evidence for the manuscript model

Five protein datasets select the model and hyperparameters through distribution and local Walsh distances.
Four further proteins test the generator without target combination fitting.
Across those four, mean scaled Wasserstein distance is 1.049 and local Walsh distance is 0.500.
Mutascape additive gives 4.501 and 0.766, respectively.
Full-dataset sequence-aware C2ST gives mean AUC 0.982, so the classifier readily detects mismatch.
Combination-calibrated Mutascape reaches 0.731 with 96 separate calibration genotypes per protein.
Both generators use all remaining experimental multi-mutant genotypes, with identical classifier folds within each protein.
The calibration set also excludes all spectral-test corners.

The corrected exploratory search study uses connected libraries from GB1, TrpB4, CreiLOV and eqFP611.
Each library contains 4,096 measured genotypes, selected through genotype coverage without consulting fitness.
WT and singles remain available before search, and model-guided methods first evaluate 32 new combinations.
All 17,280 campaigns spend their evaluation budget without random fallback.
At 96 evaluations, mean Spearman correlation is 0.730 for Mutascape and 0.824 for an additive control.
Averaging three generator families lowers mean selected-method regret from 0.0448 to 0.0186.
The paired difference interval is [-0.0902, 0.0129], so it includes no improvement.
The connected libraries favour genotypes near WT and do not establish transfer across complete landscapes.
These results support controlled synthetic benchmarks, without replacing experimentally measured combinatorial datasets.

The [singles study](singles_only_research_results.md) reports the model comparisons.
The [benchmark efficacy study](benchmark_efficacy_results_2026-09-30.md) supplies the current search results and their limits.
The [full-dataset C2ST report](full_dataset_c2st_2026-10-06.md) gives the current distribution comparison and its limits.
The [earlier calibration report](singles_pilot_extension_results.md) retains the 192-genotype experiment.
Rebuild the manuscript review from saved results with `uv run python -m scripts.build_review_manuscript`.

## Rebuild the current preprint

```bash
uv run python -m scripts.rebuild_preprint
```

This command replays saved search statistics, reruns spectral checks, rebuilds the figures, and compiles the v1 manuscript.
