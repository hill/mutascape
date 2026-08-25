# Project status

Checkpoint date is 2026-08-25.

## Publication claim

Mutascape generates synthetic combinatorial protein fitness oracles for search benchmarks. It does not predict one protein's unknown landscape.

The operational validity test is task transfer. A synthetic benchmark should preserve conclusions about which search method performs better on measured combinatorial data.

## Default model

The default is `dms-pairwise-v1`.

1. Wild type and measured single mutants define the target anchor.
2. A fitted saturating readout limits stacked effects.
3. Baseline-corrected random pair tables add shared local variation.
4. A seed fixes one lazy oracle draw.
5. The coupling remains fixed at `0.1`.

The model has no structural contact filter, donor transplant, depth residual, floor texture, or explicit higher-order term.

`0.1` is not the best coupling on the development panel. Task transfer at 48 queries peaks near `0.05`. It falls at every larger value, because the pair table has zero mean and the added variation reads as disagreement. The measured epistatic horizon points the other way and favours a value above `0.4`. Keep `0.1` and report both, since a panel that has already guided model selection cannot settle the choice.

## Current evidence

The development panel uses 16 complete SSMuLA landscapes. The external panel uses seven GraphFLA protein landscapes. Both compare 12 search methods at budgets of 48, 96, 192, and 384 queries. Each run uses ten search trials and three matched generator draws.

- At 48 queries, development rank transfer is `0.529`.
- Development winner accuracy is `0.885`.
- The development reliability pass rate is `0.875`, or 14 of 16 landscapes.
- External rank transfer is `0.439`.
- External winner accuracy is `0.864`.
- The external reliability pass rate is `1.000`, or seven of seven landscapes.

The browser keeps 96 queries as the frozen reference. Mutascape reaches `0.454` development rank transfer and `0.787` winner accuracy there. External rank transfer is `0.644`, and winner accuracy is `0.742`.

No generator leads every budget and metric. Potts and RMF remain strong controls. At 384 external queries, several rank coefficients reach one because many methods find the optimum. This is search saturation, not evidence of full landscape recovery.

The current positive claim is a low-budget decision-transfer result. The wider result shows that task validity depends on budget and decision measure.

## Known failure

The current claim covers method-ranking studies. It excludes claims about peak counts, adaptive paths, genotype values, or molecular epistasis. Walsh spectra, local optima, Wasserstein distance, and two-sample tests remain diagnostics.

## Keep

- The SSMuLA, CombinGym, and fixed GraphFLA loaders.
- The 12-method budget-matched search runner.
- Trial-level regret and paired winner outputs.
- Measured split-half reliability.
- The saturating readout and exact single-mutant anchors.
- The stable lazy oracle.
- The browser studio and task-transfer chart.
- Additive, Potts, RMF, NK, ridge, and prior-model controls.
- Walsh, local-optima, and navigability diagnostics.

## Keep as research arms

- The empirical higher-order prior.
- Donor interaction transplant.
- Site-pair scales.
- Spectral calibration.
- Structural contacts and MSA couplings.
- Floor texture and censoring policies.
- Point prediction, C2ST, MMD, and interval calibration.

These components remain useful for ablations. They are outside the default generator.

## Remove from the main claim

- Accurate genotype prediction from sparse DMS.
- Calibrated biological uncertainty from oracle draws.
- Whole-protein accuracy at high mutation depth.
- Structural realism without an independent task gain.
- One scalar distribution metric as proof of realism.
- Local-optima matching under the current coupling.

## Next experiments

1. Freeze `dms-pairwise-v1` without further tuning on the current panels.
2. Register one new source-held-out protein panel before changing the generator.
3. Increase trial counts for the 48-query and 96-query confirmatory runs.
4. Report uncertainty for panel medians and reliable winner accuracy.
5. Keep structure-aware and higher-order models outside the main paper until they improve task transfer on new data.

## Reproducibility hooks

- `scripts/run_search_transfer.py` runs the primary task test.
- `scripts/run_coupling_sensitivity.py` scores every primary metric against the coupling.
- `scripts/run_epistatic_horizon.py` compares generated and measured epistatic horizons.
- `scripts/freeze_benchmark_validity.py` checks and freezes the four-budget result.
- `scripts/run_topology_benchmark.py` runs secondary shape checks.
- `scripts/plot_benchmark_validity.py` builds the manuscript figures.
- `src/mutascape/data/validation/benchmark-validity-v1.json` drives the studio chart.
- `tests/test_benchmark_validity.py` checks the task metrics.
- `tests/test_benchmark_generators.py` checks exact anchors and lazy equivalence.

## Status flags

- `[ready]` small DMS-pairwise oracle
- `[ready]` exact lazy and enumerated agreement
- `[ready]` 16-landscape development task test
- `[ready]` seven-landscape external protein test
- `[ready]` browser visualisations
- `[ready]` v1 manuscript draft
- `[known gap]` global topology fidelity
- `[ready]` four-budget sensitivity study
- `[pending]` registered second external source
